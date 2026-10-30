"""Implicit-feedback ALS solved on CUDA, with CSR and per-row conjugate gradient.

Objective: sum c_ui (p_ui - x_u @ y_i)^2 + lambda*(||X||²+||Y||²),
c_ui=1+alpha*r_ui; p_ui=1 for observed positive interactions. Every row
update and Gram matrix is computed on the GPU. No dense ratings matrix,
unified/managed memory, or host solver is used.
"""
import os
import time

import cupy as cp
import numpy as np

KERNEL = r'''
__device__ __forceinline__ float block_sum(float value, float* sums) {
    const unsigned mask = 0xffffffff;
    for (int offset=16; offset>0; offset/=2)
        value += __shfl_down_sync(mask, value, offset);
    if (blockDim.x == 32) return __shfl_sync(mask, value, 0);
    if ((threadIdx.x & 31) == 0) sums[threadIdx.x >> 5] = value;
    __syncthreads();
    float total = sums[0] + sums[1];
    __syncthreads();
    return total;
}

extern "C" __global__ void update_rows(
    const int* ptr, const int* indices, const float* strength,
    const float* other, const float* gram, float* factors,
    int rows, int rank, float alpha, float regularization, int steps) {
    const int row=blockIdx.x, f=threadIdx.x;
    if (row>=rows) return;
    __shared__ float vector[64];
    __shared__ float sums[2];
    float x=factors[row*rank+f];
    vector[f]=x;
    __syncthreads();
    float residual=-regularization*x;
    for (int j=0;j<rank;j++) residual-=gram[f*rank+j]*vector[j];
    const int begin=ptr[row], end=ptr[row+1];
    for (int pos=begin;pos<end;pos++) {
        float y=other[indices[pos]*rank+f];
        float dot=block_sum(x*y,sums);
        float extra=alpha*strength[pos];
        residual += (1.0f+extra-extra*dot)*y;
    }
    float p=residual;
    float rr=block_sum(residual*residual,sums);
    for (int step=0;step<steps && rr>1.e-20f;step++) {
        vector[f]=p;
        __syncthreads();
        float ap=regularization*p;
        for (int j=0;j<rank;j++) ap+=gram[f*rank+j]*vector[j];
        // All threads finish reading vector before any next iteration writes it.
        __syncthreads();
        for (int pos=begin;pos<end;pos++) {
            float y=other[indices[pos]*rank+f];
            float dot=block_sum(p*y,sums);
            ap += alpha*strength[pos]*dot*y;
        }
        float pap=block_sum(p*ap,sums);
        if (!(pap>0.f)) break;
        float rate=rr/pap;
        x+=rate*p;
        residual-=rate*ap;
        float next_rr=block_sum(residual*residual,sums);
        p=residual+(next_rr/rr)*p;
        rr=next_rr;
    }
    factors[row*rank+f]=x;
}
'''


class CudaALS:
    def __init__(self, rank=32, regularization=.1, alpha=10., iterations=10,
                 seed=42, cg_steps=None):
        self.rank = int(rank)
        self.regularization = float(regularization)
        self.alpha = float(alpha)
        self.iterations = int(iterations)
        self.seed = int(seed)
        self.cg_steps = int(cg_steps or os.getenv('RETAIL_GPU_CG_STEPS', '5'))
        if self.rank not in (32, 64) or min(self.regularization, self.alpha) <= 0:
            raise ValueError('CUDA solver requires rank 32/64 and positive alpha/regularization.')
        if min(self.iterations, self.cg_steps) <= 0:
            raise ValueError('iterations and cg_steps must be positive.')
        self.user_factors = self.item_factors = None
        self._kernel = cp.RawKernel(KERNEL, 'update_rows')

    @staticmethod
    def _device_csr(matrix):
        return (cp.asarray(matrix.indptr, dtype=cp.int32),
                cp.asarray(matrix.indices, dtype=cp.int32),
                cp.asarray(matrix.data, dtype=cp.float32))

    def update(self, csr, other, factors):
        gram = other.T @ other
        self._kernel((factors.shape[0],), (self.rank,),
            (*csr, other, gram, factors, np.int32(factors.shape[0]), np.int32(self.rank),
             np.float32(self.alpha), np.float32(self.regularization), np.int32(self.cg_steps)))

    def fit(self, strengths):
        rng = np.random.default_rng(self.seed)
        self.user_factors = cp.asarray(rng.uniform(-.5/self.rank,.5/self.rank,
            (strengths.shape[0], self.rank)).astype(np.float32))
        self.item_factors = cp.asarray(rng.uniform(-.5/self.rank,.5/self.rank,
            (strengths.shape[1], self.rank)).astype(np.float32))
        users = self._device_csr(strengths)
        transposed = strengths.T.tocsr()
        items = self._device_csr(transposed)
        del transposed
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        for iteration in range(self.iterations):
            started = time.perf_counter()
            self.update(users, self.item_factors, self.user_factors)
            self.update(items, self.user_factors, self.item_factors)
            cp.cuda.Device().synchronize()
            print(f'CUDA_ALS_ITERATION {iteration+1}/{self.iterations} '
                  f'seconds={time.perf_counter()-started:.2f}', flush=True)
        del users, items
        return self

    def save(self, path):
        np.savez(path, user_factors=cp.asnumpy(self.user_factors),
            item_factors=cp.asnumpy(self.item_factors), rank=self.rank,
            regularization=self.regularization, alpha=self.alpha,
            iterations=self.iterations, seed=self.seed, cg_steps=self.cg_steps)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(**{k:data[k].item() for k in
                ['rank','regularization','alpha','iterations','seed','cg_steps']})
            model.user_factors = cp.asarray(data['user_factors'])
            model.item_factors = cp.asarray(data['item_factors'])
        return model
