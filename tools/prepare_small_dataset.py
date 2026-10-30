"""Build an isolated Retail cohort, preserving every event of selected users.

Selection uses SplitMix64(user_id) % modulus == remainder, independently of
behaviour, dates and model outcomes. Files retain their original day names and
Arrow schemas. Only the unused item embedding is omitted from the catalog.
"""
import argparse
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
DAY_US = 86_400_000_000


def selected_users(values, modulus, remainder):
    """Unsigned SplitMix64, with defined modulo-2**64 arithmetic."""
    with np.errstate(over='ignore'):
        hashed = values + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        hashed ^= hashed >> np.uint64(31)
    return hashed % np.uint64(modulus) == remainder


def prepare(source, destination, modulus, remainder):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    files = sorted((source / 'events').glob('*.pq'))
    if not files or not all(path.stem.isdigit() for path in files):
        raise ValueError('Expected events/<day>.pq files')
    parent_marker = source.parents[2] / 'retail_snapshot.json'
    parent = json.loads(parent_marker.read_text())
    if (parent_marker.parent / parent['subpath']).resolve() != source:
        raise ValueError('Source path does not match its pinned retail_snapshot.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_rows = source_bytes = output_rows = 0
    users, items = set(), set()
    actions, days = Counter(), Counter()
    event_min, event_max = None, None
    file_reports = []
    with tempfile.TemporaryDirectory(prefix='.retail-cohort-', dir=destination.parent) as temp:
        target = Path(temp) / 'snapshot'
        (target / 'events').mkdir(parents=True)
        for index, path in enumerate(files, 1):
            original = pq.ParquetFile(path)
            if original.schema_arrow.field('user_id').type != pa.uint64():
                raise ValueError('Selection requires the original uint64 user_id schema')
            before = path.stat()
            source_rows += original.metadata.num_rows
            source_bytes += before.st_size
            count = 0
            with pq.ParquetWriter(target / 'events' / path.name, original.schema_arrow,
                                  compression='zstd') as writer:
                for batch in original.iter_batches(batch_size=131_072):
                    ids = batch.column('user_id')
                    if ids.null_count:
                        raise ValueError('Null user_id: review cohort rules before reducing')
                    keep = selected_users(ids.to_numpy(zero_copy_only=False), modulus, remainder)
                    subset = batch.filter(pa.array(keep))
                    if not subset.num_rows:
                        continue
                    writer.write_batch(subset)
                    count += subset.num_rows
                    users.update(pc.unique(subset.column('user_id')).to_pylist())
                    items.update(pc.unique(subset.column('item_id')).to_pylist())
                    for row in pc.value_counts(subset.column('action_type')).to_pylist():
                        actions[row['values']] += row['counts']
                    timestamps = pc.cast(subset.column('timestamp'), pa.int64()).to_numpy()
                    lo, hi = int(timestamps.min()), int(timestamps.max())
                    event_min = lo if event_min is None else min(event_min, lo)
                    event_max = hi if event_max is None else max(event_max, hi)
                    keys, counts = np.unique(timestamps // DAY_US, return_counts=True)
                    days.update({int(k): int(n) for k, n in zip(keys, counts)})
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed while reading: {path}')
            assert pq.read_metadata(target / 'events' / path.name).num_rows == count
            output_rows += count
            file_reports.append({'name': path.name, 'source_rows': original.metadata.num_rows,
                                 'sample_rows': count, 'source_bytes': before.st_size,
                                 'source_mtime_ns': before.st_mtime_ns})
            if index % 25 == 0 or index == len(files):
                print(f'COHORT_FILES {index}/{len(files)} events={output_rows:,}', flush=True)
        if not output_rows:
            raise ValueError('Empty cohort')
        catalog_path = source / 'items.pq'
        catalog_columns = [name for name in pq.read_schema(catalog_path).names if name != 'embedding']
        catalog = pq.read_table(catalog_path, columns=catalog_columns)
        # Keep the full catalog and existing unmatched event IDs. Do not silently
        # clean or alter referential coverage during source sampling.
        pq.write_table(catalog, target / 'items.pq', compression='zstd')
        missing_items = items - set(catalog.column('item_id').to_pylist())
        output_bytes = sum(p.stat().st_size for p in target.rglob('*.pq'))
        source_bytes += catalog_path.stat().st_size
        report = {
            'created_at': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
            'source_root': str(source), 'source_revision': parent['revision'],
            'sampling': {'algorithm': 'splitmix64_uint64_v1', 'modulus': modulus,
                         'remainder': remainder, 'expected_user_fraction': 1 / modulus,
                         'unit': 'user_id', 'preserve_full_user_history': True},
            'source_events': source_rows, 'sample_events': output_rows,
            'sample_users': len(users), 'sample_event_items': len(items),
            'catalog_items': catalog.num_rows, 'event_items_missing_catalog': len(missing_items),
            'source_bytes': source_bytes, 'sample_bytes': output_bytes,
            'observed_event_fraction': output_rows / source_rows,
            'removed_catalog_columns': ['embedding'],
            'actions': dict(sorted(actions.items())), 'event_min_us': event_min,
            'event_max_us': event_max, 'first_day': min(days), 'last_day': max(days),
            'days': len(days), 'daily_events': dict(sorted(days.items())),
            'files': file_reports,
        }
        expected_days = set(range(int(files[0].stem), int(files[-1].stem) + 1))
        if set(days) != expected_days:
            raise ValueError('Cohort lost timeline coverage; choose a larger cohort')
        if set(actions) != {'view', 'click', 'added_to_cart', 'order'}:
            raise ValueError('Cohort lost an action type; choose a larger cohort')
        report['cohort_id'] = hashlib.sha256(json.dumps({
            'revision': parent['revision'], 'sampling': report['sampling'],
            'files': file_reports}, sort_keys=True).encode()).hexdigest()
        (target / 'cohort.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        target.rename(destination)
    print(json.dumps({k: v for k, v in report.items() if k not in {'files', 'daily_events'}},
                     ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT / 't_ecd_data/dataset/small/retail')
    parser.add_argument('--destination', type=Path, default=PROJECT / 't_ecd_data/reduced/retail_users025')
    parser.add_argument('--modulus', type=int, default=40)
    parser.add_argument('--remainder', type=int, default=0)
    args = parser.parse_args()
    if args.modulus < 1 or not 0 <= args.remainder < args.modulus:
        parser.error('Require modulus >= 1 and 0 <= remainder < modulus')
    prepare(args.source, args.destination, args.modulus, args.remainder)
