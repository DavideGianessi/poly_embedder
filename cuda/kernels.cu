#include <stdint.h>
#include "field.cuh"
extern "C" __global__ void sum_polynomials(
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

extern "C" __global__ void init_vanishing(
    const uint32_t* points_x,
    uint32_t* output_array,
    int n_points
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id<n_points) {
        output_array[2*id] = fe_neg(points_x[id]);
        output_array[2*id+1] = 1;
    } else {
        output_array[2*id] = 1;
        output_array[2*id+1] = 0;
    }
}

extern "C" __global__ void generate_vanishing(
    const uint32_t* input_array,
    uint32_t* output_array,
    int level
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int k = id & ((1<<(level+2))-1);
    int input_1 = id - k;
    int input_2 = input_1 + (1<<(level+1));
    int last_relevant_input= 1<<level;
    
    uint32_t sum = 0;
    int i_start = max(0, k - last_relevant_input);
    int i_end = min(k, last_relevant_input);
    
    for (int i = i_start; i <= i_end; i++) {
        int j = k - i;
        sum = fe_add(sum, fe_mul(input_array[input_1 + i], 
                 input_array[input_2 + j]));
    }
    output_array[id] = sum;
}
