#include <iostream>
#include <math.h>

// the __global__ keyword changes the function to a CUDA Kernel
__global__
void add(int n, float *x, float *y)
{
    // index of the current thread within it's block
    int index = blockIdx.x * blockDim.x * threadIdx.x;
    // number of threads in the block
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;

    // create and allocate the memory
    // this is called Unified Memory - accessible from CPU or GPU
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
	y[i] = 2.0f;
    }

    // Run kernal of 1M elements on the gpu
    // 1.
    // 2. Number of threads in a thread block
    int blocksize = 256;
    int numblocks = (N + blocksize - 1) / blocksize;
    add<<<numblocks, blocksize>>(N, x, y);

    // wait for gpu to finish before accessing on host
    cudaDeviceSynchronize();

    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}