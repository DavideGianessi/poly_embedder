#pragma once
#include <cstdint>

__constant__ uint32_t P = 4194304001u;
__constant__ uint64_t MAGIC = 4398046510ull;

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

