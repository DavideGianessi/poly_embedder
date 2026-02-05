#include "field.cuh"
extern "C" __global__ void sum_polynomials(
    const uint32_t* poly1, int size1,
    const uint32_t* poly2, int size2,
    uint32_t* result, int result_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= result_size) return;

    uint32_t val = 0;
    if (tid < size1) {
        val = fe_add(val, poly1[tid]);
    }
    if (tid < size2) {
        val = fe_add(val, poly2[tid]);
    }
    result[tid] = val;
}
