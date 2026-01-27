#include <stdint.h>
#include "field.cuh"
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
extern "C" __global__ void compute_twiddles(
    uint32_t* twiddles,
    uint32_t n,
    uint32_t base_root
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n / 2) {
        twiddles[tid] = fe_pow(base_root, tid);
    }
}
extern "C" __global__ void intt_scale(
    uint32_t* data,
    uint32_t n,
    uint32_t n_inv
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = fe_mul(data[tid], n_inv);
    }
}
extern "C" __global__ void pointwise_multiplication(
    const uint32_t* poly1,
    const uint32_t* poly2,
    int n,
    uint32_t* result
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    result[idx] = fe_mul(poly1[idx], poly2[idx]);
}
__device__ uint32_t reverse_bits(uint32_t x, uint32_t bits) {
    uint32_t res = 0;
    for (int i = 0; i < bits; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}
extern "C" __global__ void bit_reverse(uint32_t* data, uint32_t n, uint32_t bits) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t j = reverse_bits(i, bits);
    if (i < j) {
        uint32_t temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}
extern "C" __global__ void ntt_step(
    uint32_t* poly,
    const uint32_t* twiddles,
    uint32_t n,
    uint32_t len,
    uint32_t half
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;

    uint32_t block_idx = tid / half;
    uint32_t j = tid % half;
    
    uint32_t i = block_idx * len;
    uint32_t idx1 = i + j;
    uint32_t idx2 = i + j + half;

    uint32_t twiddle_idx = j * (n / len);
    uint32_t w = twiddles[twiddle_idx];

    uint32_t u = poly[idx1];
    uint32_t v = fe_mul(poly[idx2], w);

    poly[idx1] = fe_add(u, v);
    poly[idx2] = fe_sub(u, v);
}
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

#define ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define QR(a, b, c, d) ( \
    a += b, d ^= a, d = ROTL(d, 16), \
    c += d, b ^= c, b = ROTL(b, 12), \
    a += b, d ^= a, d = ROTL(d, 8), \
    c += d, b ^= c, b = ROTL(b, 7))

extern "C" __global__ void chacha20(
    uint32_t* output, 
    uint32_t n, 
    uint32_t seed0, uint32_t seed1, 
    uint32_t seed2, uint32_t seed3,
    uint32_t seed4, uint32_t seed5,
    uint32_t seed6, uint32_t seed7,
    uint32_t P
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t attempt = 0;
    uint32_t result = P;

    while (result >= P) {
        uint32_t s[16];
        s[0] = 0x61707865; s[1] = 0x33322d6e; s[2] = 0x79622d32; s[3] = 0x6b206574;
        s[4] = seed0;      s[5] = seed1;      s[6] = seed2;      s[7] = seed3;
        s[8] = seed4;      s[9] = seed5;      s[10]= seed6;      s[11]= seed7;
        
        s[12]= tid;        
        s[13]= attempt;
        s[14]= 0;         
        s[15]= 0;

        uint32_t x[16];
        for (int i = 0; i < 16; i++) x[i] = s[i];

        for (int i = 0; i < 10; i++) {
            QR(x[0], x[4], x[8],  x[12]);
            QR(x[1], x[5], x[9],  x[13]);
            QR(x[2], x[6], x[10], x[14]);
            QR(x[3], x[7], x[11], x[15]);
            QR(x[0], x[5], x[10], x[15]);
            QR(x[1], x[6], x[11], x[12]);
            QR(x[2], x[7], x[8],  x[13]);
            QR(x[3], x[4], x[9],  x[14]);
        }

        for (int i = 0; i < 16; i++) x[i] += s[i];

        result = x[0];
        attempt++; 
        
        if (attempt > 100) break; 
    }

    output[tid] = result;
}
