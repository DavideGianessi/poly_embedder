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
extern "C" __global__ void compute_weights(
    const uint32_t* points_x,
    const uint32_t* points_y,
    uint32_t* weights,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x_i = points_x[i];
    uint32_t denom = 1;
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        uint32_t diff = fe_sub(x_i, points_x[j]);
        denom = fe_mul(denom, diff);
    }
    weights[i] = fe_mul(points_y[i], fe_inv(denom));
}
extern "C" __global__ void lagrange_contribution_batched(
    const uint32_t* vanishing_poly, 
    const uint32_t* points_x,       
    const uint32_t* weights,        
    uint32_t* workspaces,
    int n,
    int points_per_thread,
    int num_workspaces
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_workspaces) return;
    uint32_t* my_poly = &workspaces[tid * n];
    for (int j = 0; j < n; j++) {
        my_poly[j] = 0;
    }
    for (int p = 0; p < points_per_thread; p++) {
        int point_idx = tid * points_per_thread + p;
        
        if (point_idx >= n) return;

        uint32_t root = points_x[point_idx];
        uint32_t w_i = weights[point_idx];

        uint32_t current = vanishing_poly[n]; 
        for (int j = n; j >= 1; j--) {
            uint32_t contribution = fe_mul(current, w_i);
            my_poly[j-1] = fe_add(my_poly[j-1], contribution);
            if (j>0) {
                current = fe_add(vanishing_poly[j-1], fe_mul(current, root));
            }
        }
    }
}
extern "C" __global__ void sum_workspaces(
    const uint32_t* workspaces, 
    uint32_t* final_result,     
    int n,
    int num_workspaces
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    uint32_t sum = 0;
    for (int i = 0; i < num_workspaces; i++) {
        sum = fe_add(sum, workspaces[i * n + j]);
    }
    final_result[j] = sum;
}
