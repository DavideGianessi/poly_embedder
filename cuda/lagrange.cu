#include "field.cuh"
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
extern "C" __global__ void lagrange_contribution_systolic(
    const uint32_t* __restrict__ vanishing_poly, 
    const uint32_t* __restrict__ points_x,       
    const uint32_t* __restrict__ weights,        
    uint32_t* workspaces,
    int n,
    int n_padded,
    int num_workspaces
) {
    const int lane = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * (blockDim.x / 32)) + (threadIdx.x / 32);
    if (warp_id >= num_workspaces) return;
    uint32_t* my_out = &workspaces[warp_id * n_padded];

    const int point_idx = warp_id * 32 + lane;

    uint32_t root = (point_idx < n) ? points_x[point_idx] : 0;
    uint32_t weight = (point_idx < n) ? weights[point_idx] : 0;

    uint32_t current = (n < n_padded) ? 0: vanishing_poly[n_padded];

    uint32_t acc = 0;
    uint32_t ready_acc = 0;
    uint32_t vanish_coeff = 0;
    int first_coeff_idx = n_padded - 32 + ((lane+31)&31);
    if (first_coeff_idx <= n) {
        vanish_coeff = vanishing_poly[first_coeff_idx];
    }

    for (int i = 0; i < 32; i++) {
        if (lane <= i) {
            acc = fe_add(acc, fe_mul(current, weight));
            current = fe_add(vanish_coeff, fe_mul(current,root));
        }
        acc       = __shfl_sync(0xffffffff, acc,       (lane + 31) & 31);
        vanish_coeff    = __shfl_sync(0xffffffff, vanish_coeff,    (lane + 31) & 31);
    }

    for (int r = 1; 32 * r  < n_padded; r++) {
        uint32_t next_vanish_coeff = vanishing_poly[n_padded - 32*(r+1) + ((lane+31)&31)];

        for (int i = 0; i < 32; i++) {
            if (lane == 0) {
                ready_acc = acc;
                acc = 0;
                vanish_coeff = next_vanish_coeff;
            }

            acc = fe_add(acc, fe_mul(current, weight));
            current = fe_add(vanish_coeff, fe_mul(current, root));

            acc       = __shfl_sync(0xffffffff, acc,       (lane + 31) & 31);
            ready_acc       = __shfl_sync(0xffffffff, ready_acc,       (lane + 31) & 31);
            vanish_coeff    = __shfl_sync(0xffffffff, vanish_coeff,    (lane + 31) & 31);
            next_vanish_coeff    = __shfl_sync(0xffffffff, next_vanish_coeff,    (lane + 31) & 31);
        }

        int write_idx = n_padded - (32 * (r - 1) + ((32 - lane) & 31)) - 1;
        if (write_idx >= 0 && write_idx < n_padded) {
            my_out[write_idx] = ready_acc;
        }
    }

    int last_r = n_padded / 32;
    for (int i = 0; i < 32; i++) {
        if (lane == 0) {
            ready_acc = acc;
        }
        
        if (lane > i) {
            acc = fe_add(acc, fe_mul(current, weight));
            current = fe_add(vanish_coeff, fe_mul(current, root));
        }

        acc       = __shfl_sync(0xffffffff, acc,       (lane + 31) & 31);
        ready_acc       = __shfl_sync(0xffffffff, ready_acc,       (lane + 31) & 31);
        vanish_coeff    = __shfl_sync(0xffffffff, vanish_coeff,    (lane + 31) & 31);
    }

    int final_write_idx = n_padded - (32 * (last_r - 1) + ((32 - lane) & 31)) - 1;
    if (final_write_idx >= 0 && final_write_idx < n_padded) {
        my_out[final_write_idx] = ready_acc;
    }
}
extern "C" __global__ void sum_workspaces(
    const uint32_t* workspaces, 
    uint32_t* final_result,     
    int n,
    int n_padded,
    int num_workspaces
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    uint32_t sum = 0;
    for (int i = 0; i < num_workspaces; i++) {
        sum = fe_add(sum, workspaces[i * n_padded + j]);
    }
    final_result[j] = sum;
}
