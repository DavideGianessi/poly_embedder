#include "field.cuh"
extern "C" __global__ void compute_twiddles(
    uint32_t* twiddles,
    uint32_t n,
    uint32_t log_n,
    uint32_t twiddle_size,
    uint32_t log_twiddle,
    bool inverse
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint32_t leading_ones = __clz(~(tid << (32 - log_n))); 
    uint32_t s = log_n - leading_ones;
    if (s < 1) return;
    uint32_t mask = (1 << (s - 1)) - 1;
    uint32_t j = tid & mask;
    uint32_t write_idx = (s - 1) * twiddle_size + j;
    uint32_t level_root = inverse ? IGENS[s] : GENS[s];
    twiddles[write_idx] = fe_pow(level_root, j);
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
extern "C" __global__ void bit_reverse_staggered(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t log_n
) {
    __shared__ uint32_t shm[32][64];
    const uint32_t tid = threadIdx.x;
    const uint32_t i = tid >> 5;
    const uint32_t j = tid & 31;
    
    const uint32_t mbits = blockIdx.x;
    const uint32_t mbits_count = log_n - 10;
    
    const uint32_t high_shift = log_n - 5;
    const uint32_t mbits_shift = 5;

    uint32_t load_addr = (i << high_shift) | (mbits << mbits_shift) | j;
    uint32_t val = input[load_addr];

    shm[j][i + j] = val;

    __syncthreads();

    uint32_t transposed_val = shm[i][j + i];

    uint32_t rev_i = __brev(i) >> (32 - 5);
    uint32_t rev_j = __brev(j) >> (32 - 5);
    uint32_t rev_mbits = __brev(mbits) >> (32 - mbits_count);

    uint32_t write_addr = (rev_i << high_shift) | (rev_mbits << mbits_shift) | rev_j;

    output[write_addr] = transposed_val;
}
extern "C" __global__ void ntt_block(
    uint32_t* data,
    const uint32_t* twiddles,
    uint32_t n,
    uint32_t twiddle_size
) {
    __shared__ uint32_t shm[2048];

    const uint32_t tid = threadIdx.x;
    const uint32_t block_start = blockIdx.x << 11;

    shm[tid] = data[block_start + tid];
    shm[tid + 1024] = data[block_start + tid + 1024];

    __syncthreads();

    #pragma unroll
    for (uint32_t s = 1; s <= 11; s++) {
        const uint32_t half = 1 << (s - 1);
        const uint32_t log_half = s - 1;
        const uint32_t twiddle_level_offset = (s - 1) * twiddle_size;

        const uint32_t block_idx = tid >> log_half;
        const uint32_t j = tid & (half - 1);

        const uint32_t idx1 = (block_idx << s) + j;
        const uint32_t idx2 = idx1 + half;

        const uint32_t w = twiddles[twiddle_level_offset + j];

        const uint32_t u = shm[idx1];
        const uint32_t v = fe_mul(shm[idx2], w);

        shm[idx1] = fe_add(u, v);
        shm[idx2] = fe_sub(u, v);

        __syncthreads();
    }

    data[block_start + tid] = shm[tid];
    data[block_start + tid + 1024] = shm[tid + 1024];
}
extern "C" __global__ void ntt_step(
    uint32_t* poly,
    const uint32_t* twiddles,
    uint32_t n,
    uint32_t log_len,
    uint32_t len,
    uint32_t log_half,
    uint32_t half,
    uint32_t twiddle_offset
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (n >> 1)) return;
    uint32_t block_idx = tid >> log_half;
    uint32_t j = tid & (half - 1);
    uint32_t i = block_idx << log_len;
    uint32_t idx1 = i + j;
    uint32_t idx2 = idx1 + half;
    uint32_t w = twiddles[twiddle_offset + j];
    uint32_t u = poly[idx1];
    uint32_t v = fe_mul(poly[idx2], w);

    poly[idx1] = fe_add(u, v);
    poly[idx2] = fe_sub(u, v);
}
