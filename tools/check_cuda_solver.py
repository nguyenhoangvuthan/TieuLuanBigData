"""Compare both CUDA block widths against independently built normal equations."""
from runtime import managed_entrypoint

managed_entrypoint()
import os
import cupy as cp
import numpy as np
from scipy import sparse

from cuda_als import CudaALS

strengths = sparse.csr_matrix(np.array([[2,0,1,0,0,3,0], [0,1,0,2,4,0,1],
                                       [0,0,0,0,0,0,0], [1,2,1,2,1,2,1]], np.float32))
for rank in [32,64]:
    rng=np.random.default_rng(42)
    other=rng.normal(0,.2,(7,rank)).astype(np.float32)
    initial=rng.normal(0,.2,(4,rank)).astype(np.float32)
    model=CudaALS(rank=rank,regularization=.3,alpha=10,iterations=1,cg_steps=64)
    factors=cp.asarray(initial)
    model.update(model._device_csr(strengths),cp.asarray(other),factors)
    cp.cuda.Device().synchronize()
    actual=cp.asnumpy(factors)
    expected=[]
    for row in range(4):
        ids=strengths[row].indices
        r=strengths[row].data.astype(np.float64)
        y=other[ids].astype(np.float64)
        a=other.astype(np.float64).T@other + .3*np.eye(rank) + y.T@(10*r[:,None]*y)
        b=y.T@(1+10*r)
        expected.append(np.linalg.solve(a,b))
    np.testing.assert_allclose(actual,expected,atol=3e-4,rtol=3e-3)
    # Production's truncated CG must still reduce each fixed-item objective.
    model.cg_steps=5
    factors=cp.asarray(initial)
    model.update(model._device_csr(strengths),cp.asarray(other),factors)
    actual=cp.asnumpy(factors)
    for row in range(4):
        ids=strengths[row].indices;r=strengths[row].data.astype(np.float64)
        y=other[ids].astype(np.float64)
        a=other.astype(np.float64).T@other+.3*np.eye(rank)+y.T@(10*r[:,None]*y)
        b=y.T@(1+10*r)
        objective=lambda x: x@a@x-2*b@x
        assert objective(actual[row])<=objective(initial[row])+1e-4
    print(f'CUDA_NORMAL_EQUATIONS_AND_DESCENT_RANK_{rank}_PASS',flush=True)
from gpu_als import GpuALSModel

os.environ['RETAIL_GPU_USER_BATCH']='2'
os.environ['RETAIL_GPU_ITEM_BATCH']='2'
seen=sparse.csr_matrix([[1,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1]],dtype=np.float32)
x=np.array([[1,2],[0,1],[-1,1]],np.float32)
y=np.array([[3,1],[2,2],[1,3],[4,0],[1,0],[0,3]],np.float32)
scorer=GpuALSModel(None,seen,{}, {})
scorer._factors=(cp.asarray(x),cp.asarray(y))
for novel in [False,True]:
    scores=x@y.T
    if novel:scores[seen.toarray().astype(bool)]=-np.inf
    expected={u:[int(i) for i in np.lexsort((np.arange(6),-scores[u]))[:3]
                 if np.isfinite(scores[u,i])] for u in range(3)}
    assert dict(scorer.recommend_indices([0,1,2],k=3,novel=novel))==expected
scorer.close()
print('GPU_TILED_TOPK_MATCHES_NUMPY_FULL_RANKING_PASS',flush=True)
print('ALL_CUDA_SOLVER_CHECKS_PASSED',flush=True)
