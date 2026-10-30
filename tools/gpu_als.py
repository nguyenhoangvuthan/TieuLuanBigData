"""CUDA ALS and exact, bounded-memory Top-N for the Retail Spark notebook.

Spark handles ETL. Only aggregated triples cross into Python, through local
Parquet batches rather than collect()/toPandas(). The CUDA solver uses constant L2
(not Spark ALS-WR); parameters are selected on validation for this backend.
"""
import gc
import ctypes
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from scipy import sparse


def gpu_info():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props['name']
    return {'backend': 'cupy_cuda_cg', 'gpu': name.decode() if isinstance(name, bytes) else name,
            'vram_total_bytes': int(total), 'vram_free_bytes': int(free),
            'confidence': '1 + alpha * interaction_strength',
            'regularization': 'constant L2; distinct from Spark ALS-WR',
            'solver': 'CUDA conjugate gradient',
            'cg_steps': int(os.getenv('RETAIL_GPU_CG_STEPS', '5'))}


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def evaluation_population(spark, labels, user_map, item_map, *, seen=None,
                          novel=False, cold_only=False):
    """Evaluate the small label set against the exact fitted ID maps/CSR.

    This avoids repeatedly scanning 240M raw history events for quantities
    already represented by the maps and sparse interaction matrix.
    """
    users = {row.user_id: int(row.user_index) for row in user_map.toLocalIterator()}
    items = {row.item_id: int(row.item_index) for row in item_map.toLocalIterator()}
    grouped = {}
    for row in labels.select('user_id', 'item_id').toLocalIterator():
        grouped.setdefault(row.user_id, set()).add(row.item_id)
    cold = set(grouped).difference(users)
    eligible = cold if cold_only else set(grouped).intersection(users)
    if novel and seen is None:
        raise ValueError('Novel labels require the fitted history CSR.')
    truth = []
    label_items = set()
    new_pairs = positive_pairs = 0
    for user in sorted(eligible):
        relevant = grouped[user]
        if novel and user in users:
            uid = users[user]
            known = seen.indices[seen.indptr[uid]:seen.indptr[uid + 1]]
            item_labels = sorted(relevant)
            encoded = np.asarray([items.get(item, -1) for item in item_labels], np.int32)
            positions = np.searchsorted(known, encoded)
            matched = np.zeros(len(encoded), dtype=bool)
            within = positions < len(known)
            matched[within] = known[positions[within]] == encoded[within]
            relevant = {item for item, present in zip(item_labels, matched) if not present}
        if relevant:
            truth.append((user, sorted(relevant)))
            positive_pairs += len(relevant)
            new_pairs += len(relevant.difference(items))
            label_items.update(relevant)
    report = {'positive_users': len(grouped), 'cold_users': len(cold),
              'eligible_users': len(truth), 'removed_no_novel_labels': len(eligible) - len(truth),
              'positive_pairs': positive_pairs, 'new_item_positive_pairs': new_pairs,
              'label_items': len(label_items), 'new_label_items': len(label_items.difference(items))}
    report.update(cold_user_rate=len(cold) / len(grouped) if grouped else None,
                  new_item_pair_rate=new_pairs / positive_pairs if positive_pairs else None,
                  new_item_distinct_rate=report['new_label_items'] / len(label_items) if label_items else None)
    return spark.createDataFrame(truth, 'user_id string, labels array<string>'), report


def dataframe_to_csr(pairs, cache_dir, host_budget_bytes):
    """Export and stream triples; retain at most one Arrow batch plus CSR buffers."""
    plan = pairs._jdf.queryExecution().optimizedPlan().canonicalized().toString()
    signature = hashlib.sha256((plan + '\n' + '\n'.join(sorted(pairs.inputFiles()))).encode()).hexdigest()
    directory = Path(cache_dir) / signature
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / 'strengths.npz'
    metadata_path = directory / 'metadata.json'
    if matrix_path.exists() and metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta['peak_host_estimate_bytes'] > host_budget_bytes:
            raise MemoryError('Cached CSR exceeds the configured host RAM budget.')
        return sparse.load_npz(matrix_path), meta
    parquet_path = directory / 'triples'
    if not (parquet_path / '_SUCCESS').exists():
        from pyspark.sql import functions as F
        (pairs.select(F.col('user_index').cast('int'), F.col('item_index').cast('int'),
                      F.col('interaction_strength').cast('float'))
         .write.mode('overwrite').parquet(str(parquet_path.resolve())))
    dataset = ds.dataset(parquet_path, format='parquet', exclude_invalid_files=True)
    count = dataset.count_rows()
    if not count or count >= np.iinfo(np.int32).max:
        raise ValueError('GPU CSR requires a nonempty input with < 2^31 interactions.')
    # COO, CSR, confidence data, host transpose and Arrow scratch space.
    peak_estimate = count * 36 + 128 * 1024**2
    if peak_estimate > host_budget_bytes:
        raise MemoryError(f'CSR needs about {peak_estimate / 1024**3:.2f} GiB; '
                          'raise RETAIL_GPU_HOST_BUDGET_GB only if physical RAM is available.')
    rows = np.empty(count, np.int32)
    cols = np.empty(count, np.int32)
    values = np.empty(count, np.float32)
    offset = 0
    for batch in dataset.to_batches(batch_size=65536, use_threads=False):
        size = batch.num_rows
        for name, target in [('user_index', rows), ('item_index', cols),
                             ('interaction_strength', values)]:
            column = batch.column(name)
            if column.null_count:
                raise ValueError(f'Nulls in {name}')
            target[offset:offset + size] = column.to_numpy(zero_copy_only=False)
        offset += size
    if offset != count or np.any(rows < 0) or np.any(cols < 0):
        raise ValueError('Invalid interaction indices/count.')
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Implicit interactions must be finite and strictly positive.')
    shape = (int(rows.max()) + 1, int(cols.max()) + 1)
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
    del rows, cols, values, dataset
    gc.collect()
    matrix.sort_indices()
    temporary = directory / 'strengths.tmp.npz'
    sparse.save_npz(temporary, matrix, compressed=False)
    temporary.replace(matrix_path)
    meta = {'signature': signature, 'shape': list(shape), 'nnz': int(matrix.nnz),
            'peak_host_estimate_bytes': peak_estimate,
            'matrix_path': str(matrix_path.resolve())}
    _atomic_json(metadata_path, meta)
    shutil.rmtree(parquet_path)
    return matrix, meta


class GpuALSModel:
    def __init__(self, model, strengths, params, metadata):
        self.model = model
        self.strengths = strengths
        self.params = params
        self.metadata = metadata
        self._factors = None

    @classmethod
    def fit(cls, pairs, params, cache_dir, *, iterations=10, seed=42):
        import cupy as cp
        try:
            from .cuda_als import CudaALS
        except ImportError:
            from cuda_als import CudaALS
        gpu_info()  # Fail explicitly; never silently fit on CPU.
        started = time.perf_counter()
        budget = float(os.getenv('RETAIL_GPU_HOST_BUDGET_GB', '2.5')) * 1024**3
        strengths, metadata = dataframe_to_csr(pairs, cache_dir, budget)
        free, _ = cp.cuda.runtime.memGetInfo()
        csr_bytes = strengths.data.nbytes + strengths.indices.nbytes + strengths.indptr.nbytes
        estimated_gpu = 2 * csr_bytes + sum(strengths.shape) * int(params['rank']) * 4 + 512 * 1024**2
        if estimated_gpu > free * .85:
            raise MemoryError('The sparse training matrices would exceed the available GPU VRAM.')
        model = CudaALS(rank=int(params['rank']),
            regularization=float(params['regParam']), alpha=float(params['alpha']),
            iterations=int(iterations), seed=int(seed))
        prepare_seconds = time.perf_counter() - started
        fit_started = time.perf_counter()
        model.fit(strengths)
        cp.cuda.Device().synchronize()
        fit_seconds = time.perf_counter() - fit_started
        if not all(bool(cp.isfinite(factors).all()) for factors in
                   [model.user_factors, model.item_factors]):
            raise FloatingPointError('CUDA ALS produced non-finite factors.')
        metadata = dict(metadata, prepare_seconds=prepare_seconds, cuda_fit_seconds=fit_seconds,
                        iterations=int(iterations), seed=int(seed), backend='cupy_cuda_cg', cg_steps=model.cg_steps,
                        confidence='1 + alpha * strength', regularization='constant_l2')
        print(f'GPU_ALS_FIT shape={strengths.shape} nnz={strengths.nnz:,} '
              f'prepare={prepare_seconds:.1f}s cuda={fit_seconds:.1f}s', flush=True)
        return cls(model, strengths, dict(params), metadata), time.perf_counter() - started

    def _device_factors(self):
        if self._factors is None:
            self._factors = (self.model.user_factors, self.model.item_factors)
        return self._factors

    @staticmethod
    def _order(ids, scores, k):
        import cupy as cp
        # CuPy lexsort has no per-row axis; two stable GPU sorts implement
        # score descending with item ID ascending as the secondary key.
        by_id = cp.argsort(ids, axis=1, kind='stable')
        values = cp.take_along_axis(scores, by_id, axis=1)
        by_score = cp.argsort(-values, axis=1, kind='stable')[:, :k]
        return cp.take_along_axis(by_id, by_score, axis=1)

    @staticmethod
    def _topk(scores, item_ids, k):
        """Exact score-desc/id-asc selection, including ties at the kth score."""
        import cupy as cp
        k = min(k, scores.shape[1])
        candidates = cp.argpartition(-scores, k - 1, axis=1)[:, :k]
        selected_scores = cp.take_along_axis(scores, candidates, axis=1)
        threshold = selected_scores.min(axis=1, keepdims=True)
        selected_ids = cp.take_along_axis(cp.broadcast_to(item_ids, scores.shape), candidates, axis=1)
        strict = selected_scores > threshold
        maximum = np.iinfo(np.int32).max
        # At most k-1 strictly better scores; fill the boundary with lowest IDs.
        tied_ids = cp.where(scores == threshold, cp.broadcast_to(item_ids, scores.shape), maximum)
        boundary = cp.argpartition(tied_ids, k - 1, axis=1)[:, :k]
        boundary_ids = cp.take_along_axis(tied_ids, boundary, axis=1)
        ids = cp.concatenate([cp.where(strict, selected_ids, maximum), boundary_ids], axis=1)
        vals = cp.concatenate([cp.where(strict, selected_scores, -cp.inf),
                               cp.where(boundary_ids != maximum, threshold, -cp.inf)], axis=1)
        order = GpuALSModel._order(ids, vals, k)
        return cp.take_along_axis(ids, order, axis=1), cp.take_along_axis(vals, order, axis=1)

    def recommend_indices(self, user_ids, k=10, novel=False, scored=False, seen=None):
        """Score bounded user/item tiles; never materialize users x all-items.

        Yields (user, items), or (user, items, scores) when scored=True.
        With novel=True, items in `seen` (default: the fitted CSR) are excluded;
        pass the full-history CSR when the fitted matrix was reduced.
        """
        history = self.strengths if seen is None else seen
        import cupy as cp
        user_factors, item_factors = self._device_factors()
        user_batch = int(os.getenv('RETAIL_GPU_USER_BATCH', '64'))
        item_batch = int(os.getenv('RETAIL_GPU_ITEM_BATCH', '8192'))
        if min(user_batch, item_batch, k) <= 0:
            raise ValueError('GPU batch sizes and k must be positive.')
        for begin in range(0, len(user_ids), user_batch):
            ids = np.asarray(user_ids[begin:begin + user_batch], np.int32)
            queries = user_factors[cp.asarray(ids)]
            best_ids = cp.empty((len(ids), 0), dtype=cp.int32)
            best_scores = cp.empty((len(ids), 0), dtype=cp.float32)
            known = history[ids] if novel else None
            for first in range(0, item_factors.shape[0], item_batch):
                end = min(first + item_batch, item_factors.shape[0])
                scores = queries @ item_factors[first:end].T
                if known is not None:
                    # Only transfer sparse seen coordinates belonging to this tile.
                    coordinates = known[:, first:end].tocoo()
                    scores[cp.asarray(coordinates.row), cp.asarray(coordinates.col)] = -cp.inf
                tile_ids = cp.arange(first, end, dtype=cp.int32)[None, :]
                tile_ids, tile_scores = self._topk(scores, tile_ids, k)
                candidate_ids = cp.concatenate([best_ids, tile_ids], axis=1)
                candidate_scores = cp.concatenate([best_scores, tile_scores], axis=1)
                order = self._order(candidate_ids, candidate_scores, k)
                best_ids = cp.take_along_axis(candidate_ids, order, axis=1)
                best_scores = cp.take_along_axis(candidate_scores, order, axis=1)
            host_ids, host_scores = cp.asnumpy(best_ids), cp.asnumpy(best_scores)
            for user, recs, scores in zip(ids, host_ids, host_scores):
                kept = [(int(item), float(score)) for item, score in zip(recs, scores)
                        if np.isfinite(score) and item < self.strengths.shape[1]]
                if scored:
                    yield int(user), [item for item, _ in kept], [score for _, score in kept]
                else:
                    yield int(user), [item for item, _ in kept]

    def recommend_spark(self, spark, users, user_map, item_map, k=10, novel=False):
        # These are ID maps and evaluation populations, not interaction tables.
        indexed = users.select('user_id').distinct().join(user_map, 'user_id')
        ids = {int(row.user_index): row.user_id for row in indexed.toLocalIterator()}
        items = {int(row.item_index): row.item_id for row in item_map.toLocalIterator()}
        if len(items) != self.strengths.shape[1]:
            raise ValueError('Recommendation item map differs from the fitted model.')
        result = [(ids[user], [items[item] for item in recs]) for user, recs in
                  self.recommend_indices(sorted(ids), k, novel)]
        return spark.createDataFrame(result, 'user_id string, recommendations array<string>')

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Model already exists: {path}')
        staging = Path(tempfile.mkdtemp(prefix=path.name + '-', dir=path.parent))
        try:
            self.model.save(staging / 'model.npz')
            sparse.save_npz(staging / 'strengths.npz', self.strengths, compressed=False)
            _atomic_json(staging / 'metadata.json', {'params': self.params, **self.metadata})
            staging.rename(path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @classmethod
    def load(cls, path):
        try:
            from .cuda_als import CudaALS
        except ImportError:
            from cuda_als import CudaALS
        path = Path(path)
        metadata = json.loads((path / 'metadata.json').read_text())
        if metadata['peak_host_estimate_bytes'] > float(os.getenv('RETAIL_GPU_HOST_BUDGET_GB', '2.5')) * 1024**3:
            raise MemoryError('Saved model exceeds the configured host RAM budget.')
        return cls(CudaALS.load(path / 'model.npz'),
                   sparse.load_npz(path / 'strengths.npz'), metadata['params'], metadata)

    def close(self):
        import cupy as cp
        self._factors = None
        self.model = None
        self.strengths = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
        # NumPy/Arrow buffers may be freed but retained in glibc arenas.
        libc = ctypes.CDLL(None)
        if hasattr(libc, 'malloc_trim'):
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            libc.malloc_trim(0)
