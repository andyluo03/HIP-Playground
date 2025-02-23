#include <cuda_runtime.h>

#define kTileSize 16

__global__ 
void deviceGemm(int m, int n, int k, const int8_t *A, const int8_t *B, int32_t *C) {
    __shared__ int8_t sharedA[kTileSize][kTileSize + 1];
    __shared__ int8_t sharedB[kTileSize][kTileSize + 1];
    
    int c_i = blockIdx.x * blockDim.x + threadIdx.x;
    int c_j = blockIdx.y * blockDim.y + threadIdx.y;

    int32_t result = 0;

    for (int i = 0; i < k; i += kTileSize) {
        // Load A, B into shared.
        int targetA_i = c_i + i;
        int targetA_j = c_j;

        int targetB_i = c_i;
        int targetB_j = c_j + i;

        if (targetA_i < m && targetA_j < k) {
            sharedA[threadIdx.x][threadIdx.y] = A[targetA_i * k + targetA_j];
        } else {
            sharedA[threadIdx.x][threadIdx.y] = 0;
        }

        if (targetB_i < k && targetB_j < n) {
            sharedB[threadIdx.x][threadIdx.y] = B[targetB_i * n + targetB_j];
        } else {
            sharedB[threadIdx.x][threadIdx.y] = 0;
        }

        __syncthreads();

        // Calculate for tile effect. DO NOT SPECIALIZE.
        for (int j = 0; j < kTileSize; j++) {
            result += sharedA[threadIdx.x][j] * sharedB[j][threadIdx.y];
        }
        
        __syncthreads();
    }

    // Specialize for writing. 
    if (c_i < m && c_j < n) { 
        C[c_i * n + c_j] = result; 
    }
} 

void gemm(int m, int n, int k, const int8_t* A, const int8_t* B, int32_t* C) {
    // Create Dimensions
    dim3 gridDim(ceil(m / kTileSize), ceil(n / kTileSize), 1);
    dim3 blockDim(kTileSize, kTileSize, 1);

    // Launch Kernel
    deviceGemm<<<gridDim, blockDim>>>(m, n, k, A, B, C);
}

int main () {
    
}