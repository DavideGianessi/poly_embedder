#include <stdint.h>
#include "field.cuh"
extern "C" __global__ void sum_polynomials_flex_kernel(
    const uint32_t* poly1,
    int size1,
    const uint32_t* poly2,
    int size2,
    uint32_t* result,
    int result_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= result_size) return;
    
    uint32_t val1 = (idx < size1) ? poly1[idx] : 0;
    uint32_t val2 = (idx < size2) ? poly2[idx] : 0;
    
    result[idx] = fe_add(val1, val2);
}
