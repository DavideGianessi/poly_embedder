#pragma once
#include <cstdint>

__constant__ uint32_t P = 4194304001u;
__constant__ uint64_t MAGIC = 4398046510ull;
__constant__ uint32_t GENS[25] = {
    1, 4194304000, 809539273, 2303415184, 1800537630, 2906399817, 369001549, 2026377158,
    1867760616, 3185713831, 3100728574, 3986884701, 2037177755, 3682666484, 1581848693, 217320144,
    623292090, 502725452, 790764273, 1079588648, 3440443607, 1688530187, 2541931790, 2936257672,
    2580763344,
};
__constant__ uint32_t IGENS[25] = {
    1, 4194304000, 3384764728, 3412379098, 1559634102, 1560690925, 1481810193, 3824470519,
    306209204, 235196417, 402301397, 4159660757, 3602029040, 2380151834, 1885459, 2469224405,
    3336134804, 3231469334, 1976201916, 4149395070, 1476203138, 1004409423, 3013869102, 2962262218,
    3810123335,
};

__device__ __forceinline__ uint32_t fe_mul(uint32_t a, uint32_t b) {
    uint64_t x = (uint64_t)a * b;
    uint64_t q = ((x >> 32) * MAGIC) >> 32;
    uint64_t r = x - q * (uint64_t)P;
    if (r >= P) r -= P;
    if (r >= P) r -= P;
    return (uint32_t)r;
}

__device__ __forceinline__ uint32_t fe_add(uint32_t a, uint32_t b) {
    uint64_t x = (uint64_t)a + b;
    if (x >= P) x -= P;
    return (uint32_t)x;
}

__device__ __forceinline__ uint32_t fe_sub(uint32_t a, uint32_t b) {
    uint64_t x = (uint64_t)P + a - b;
    if (x >= P) x -= P;
    return (uint32_t)x;
}

__device__ __forceinline__ uint32_t fe_neg(uint32_t a) {
    uint64_t x = (uint64_t)P - a;
    if (x >= P) x -= P;
    return (uint32_t)x;
}

__device__ uint32_t fe_inv(uint32_t a) {
    uint32_t result = 1;
    uint32_t base = a;
    uint32_t exp = P - 2;
    
    while (exp > 0) {
        if (exp & 1) {
            result = fe_mul(result, base);
        }
        base = fe_mul(base, base);
        exp >>= 1;
    }
    return result;
}

__device__ uint32_t fe_pow(uint32_t base, uint32_t exp) {
    uint32_t res = 1;
    while (exp > 0) {
        if (exp & 1) res = fe_mul(res, base);
        base = fe_mul(base, base);
        exp >>= 1;
    }
    return res;
}

