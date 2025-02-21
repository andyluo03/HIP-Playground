#include <hip/hip_runtime.h>

__global__ 
void hello_from () {
    printf("Hello from block: %d and thread: %d\n", blockIdx.x, threadIdx.x);
}

int main () {
    hello_from<<<8, 8>>>();

    hipDeviceSynchronize();
}