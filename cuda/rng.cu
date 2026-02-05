#include "field.cuh"
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
