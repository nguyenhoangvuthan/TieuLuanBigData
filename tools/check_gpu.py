"""Actual CUDA regression checks, in an isolated Spark session with bounded RAM/swap."""
from runtime import managed_entrypoint

managed_entrypoint()
import json
import os
import tempfile
from pathlib import Path

import cupy as cp
from pyspark.sql import SparkSession

from gpu_als import GpuALSModel, evaluation_population, gpu_info
from runtime import memory_scope

print('GPU', json.dumps(gpu_info()), flush=True)
print('MEMORY_SCOPE', json.dumps(memory_scope()), flush=True)
spark = (SparkSession.builder.master('local[1]').appName('retail-cuda-regression')
         .config('spark.driver.memory', '768m').config('spark.ui.enabled', 'false')
         .config('spark.ui.showConsoleProgress', 'false')
         .config('spark.executor.heartbeat.dropZeroAccumulatorUpdates', 'false')
         .config('spark.sql.shuffle.partitions', '2').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
try:
    interactions = [(0, 0, 2.), (0, 1, 1.)] + [(1, item, 1.) for item in range(6)] + [(2, 5, 3.)]
    pairs = spark.createDataFrame(interactions, 'user_index int, item_index int, interaction_strength float')
    users = spark.createDataFrame([(f'u{i}', i) for i in range(3)], 'user_id string, user_index int')
    items = spark.createDataFrame([(f'i{i}', i) for i in range(6)], 'item_id string, item_index int')
    os.environ['RETAIL_GPU_USER_BATCH'] = '2'
    os.environ['RETAIL_GPU_ITEM_BATCH'] = '2'
    with tempfile.TemporaryDirectory() as temporary:
        model, seconds = GpuALSModel.fit(pairs, {'rank':32, 'regParam':.1, 'alpha':10.},
            Path(temporary)/'cache', iterations=2, seed=42)
        assert model.model.__class__.__name__ == 'CudaALS'
        assert bool(cp.isfinite(model.model.user_factors).all())
        print('CUDA_FIT_PASS', seconds, flush=True)
        expected = dict(model.recommend_indices([0,1,2], k=3))
        path = Path(temporary)/'saved'
        model.save(path)
        restored = GpuALSModel.load(path)
        assert dict(restored.recommend_indices([0,1,2], k=3)) == expected
        restored.close()
        print('GPU_MODEL_ROUNDTRIP_PASS', flush=True)
        model._factors = (cp.ones((3, 2), dtype=cp.float32), cp.ones((6, 2), dtype=cp.float32))
        assert dict(model.recommend_indices([0,1,2],k=3)) == {0:[0,1,2],1:[0,1,2],2:[0,1,2]}
        assert dict(model.recommend_indices([0,1,2],k=3,novel=True)) == {0:[2,3,4],1:[],2:[0,1,2]}
        # Lịch sử đầy đủ (ma trận fit đã rút gọn): u2 còn thấy i0 dù cặp đó không có trong CSR fit.
        full = model.strengths.tolil(); full[2, 0] = 1.
        assert dict(model.recommend_indices([0,1,2],k=3,novel=True,seen=full.tocsr())) == {0:[2,3,4],1:[],2:[1,2,3]}
        print('EXACT_GPU_TOPK_TIES_AND_NOVEL_MASK_PASS', flush=True)
        labels = spark.createDataFrame([('u0','i0'),('u0','i2'),('u0','NEW'),('u1','i0'),
                                        ('u2','i1'),('cold','i1')], 'user_id string, item_id string')
        truth,report=evaluation_population(spark,labels,users,items,seen=model.strengths)
        assert (report['positive_users'],report['cold_users'],report['eligible_users'])==(4,1,3),report
        assert report['new_item_positive_pairs']==1,report
        truth,report=evaluation_population(spark,labels,users,items,seen=model.strengths,novel=True)
        # u0 đã thấy i0, i1; u1 đã thấy mọi item; u2 chỉ thấy i5.
        assert {r.user_id:sorted(r.labels) for r in truth.collect()}=={'u0':['NEW','i2'],'u2':['i1']}
        assert report['removed_no_novel_labels']==1,report
        truth,report=evaluation_population(spark,labels,users,items,cold_only=True)
        assert [r.user_id for r in truth.collect()]==['cold'],report
        print('POPULATION_RULES_PASS',flush=True)
        model.close()
    print('ALL_GPU_CHECKS_PASSED',flush=True)
finally:
    spark.stop()
