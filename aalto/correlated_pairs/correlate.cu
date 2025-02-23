#include <iostream>

#include <cuda_runtime.h>

__device__
float stddev(int nx, int idx, const float* data) {
    float average = 0.0;
    for (int i = 0; i < nx; i++) { average += data[i + idx*nx]; }
    average /= nx;

    float variance = 0.0;
    for (int i = 0; i < nx; i++) {
        variance += (data[i + nx*idx] - average) * (data[i + nx*idx] - average);
    }
    variance /= (nx - 1);

    return sqrtf(variance);
}

__device__
float cov(int nx, int i_index, int j_index, const float* data) {
    float mean_i = 0;
    float mean_j = 0;

    for (int i = 0; i < nx; i++) {
        mean_i += data[i_index * nx + i];
        mean_j += data[j_index * nx + i];
    }
    
    mean_i /= nx;
    mean_j /= nx;

    float cov = 0;
    for (int i = 0; i < nx; i++) {
        float j_i = data[j_index * nx + i];
        float i_i = data[i_index * nx + i];

        cov += (j_i - mean_j) * (i_i - mean_i);
    }

    return cov / (nx-1);
}

__global__ 
void deviceCorrelate(int ny, int nx, const float *data, float *result) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int i_index = index / ny;
    int j_index = index % ny;

    if (i_index < j_index) { return; } // Some warps might be able to no-op
    if (i_index >= ny || j_index >= ny)     { return; }     // 
    
    // Calculate Correlation Coefficient Between i and j
    // Correlation Coefficient = cov(X, Y) / (stdev(x) * stdev(y))
    result[i_index + j_index * ny] = 
        cov(nx, i_index, j_index, data) / 
        (stddev(nx, i_index, data) * stddev(nx, j_index, data));
}

void correlate(int ny, int nx, const float *data, float *result) {
    dim3 gridDim( ceil ((ny*ny) / 32.0), 1, 1);
    dim3 blockDim(32, 1, 1);

    deviceCorrelate<<<gridDim, blockDim>>>(ny, nx, data, result);
}

int main () {
    int nx = 2;
    int ny = 2;

    float *data = (float*) malloc(sizeof(float) * nx * ny);
    float *result = (float*) malloc(sizeof(float) * ny * ny);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            data[i * nx + j] = rand();
        }
    }

    correlate(ny, nx, data, result);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < ny; j++) {
            std::cout << result[i * ny + j] << ", ";
        }
        std::cout << std::endl;
    }

    free(data);
    free(result);
}