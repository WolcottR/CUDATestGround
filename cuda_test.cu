#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>

// the __global__ keyword changes the function to a CUDA Kernel
__global__
void add(int n, float *x, float *y, float *z)
{
    // index of the current thread within it's block
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // number of threads in the block
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        z[i] = x[i] + y[i];
}

int main(void)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
	printf("cudaGetDeviceCount returned %d\n->%s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));
	exit(EXIT_FAILURE);
    }

    int dev = 0;
	
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int maxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    std::cout << "Max Threads Per Multi Processor: " << maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads Per Block: " << maxThreadsPerBlock << std::endl;

    int N = 1<<20;

    // create and allocate the memory
    // this is called Unified Memory - accessible from CPU or GPU
    float *x, *y, *z;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&z, N*sizeof(float));
    

    for (int i = 0; i < N; i++)
    {
        x[i] = 2.0f;
	y[i] = 5.0f;
    }

    // Run kernal of 1M elements on the gpu
    // 1.
    // 2. Number of threads in a thread block
    int blocksize = maxThreadsPerBlock;
    int numblocks = (N + blocksize - 1) / blocksize;
    add<<<numblocks, blocksize>>>(N, x, y, z);

    // wait for gpu to finish before accessing on host
    cudaDeviceSynchronize();

    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
	// std::cout << "Amount: " << z[i] << std::endl;
        maxError = fmax(maxError, fabs(z[i] - 7.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}