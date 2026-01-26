#pragma once
#include <cstdint>

__constant__ uint32_t P = 4194304001u;
__constant__ uint64_t MAGIC = 4398046510ull;
__constant__ uint32_t GENS[25] = {
    1, 4194304000u, 809539273u, 2303415184u, 1800537630u, 
    2906399817u, 369001549u, 2026377158u, 1867760616u, 
    3185713831u, 3100728574u, 3986884701u, 2037177755u, 
    3682666484u, 1581848693u, 217320144u, 623292090u, 
    502725452u, 790764273u, 1079588648u, 3440443607u, 
    1688530187u, 2541931790u, 2936257672u, 2580763344u
};

__constant__ uint32_t IGENS[25] = {
    1, 4194304000u, 3384764728u, 3412379098u, 1559634102u, 
    1560690925u, 1481810193u, 3824470519u, 306209204u, 
    235196417u, 402301397u, 4159660757u, 3602029040u, 
    2380151834u, 1885459u, 2469224405u, 3336134804u, 
    3231469334u, 1976201916u, 4149395070u, 1476203138u, 
    1004409423u, 3013869102u, 2962262218u, 3810123335u
};

__constant__ uint32_t N_INVS[25] = {
    1, 2097152001u, 3145728001u, 3670016001u, 3932160001u, 
    4063232001u, 4128768001u, 4161536001u, 4177920001u, 
    4186112001u, 4190208001u, 4192256001u, 4193280001u, 
    4193792001u, 4194048001u, 4194176001u, 4194240001u, 
    4194272001u, 4194288001u, 4194296001u, 4194300001u, 
    4194302001u, 4194303001u, 4194303501u, 4194303751u
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


__device__ uint32_t get_root_of_unity(int order, bool inverse) {
    if (inverse) {
        return IGENS[order];
    } else {
        return GENS[order];
    }
}

__device__ uint32_t get_n_inv(int order) {
    return N_INVS[order];
}
