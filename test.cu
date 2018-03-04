#include <iostream>
#include <math.h>

// the __global__ keyword changes the function to a CUDA Kernel
__global__
void add(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;

    // create and allocate the memory
    std::cout << "Pre-Allocation" << std::endl;
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    std::cout << "Post-Allocation" << std::endl;

    for (int i = 0; i < N; i++)
    {
        std::cout << "inside loop" << std::endl;
        x[i] = 1.0f;
	y[i] = 2.0f;
    }

    std::cout << "Entry Into Add" << std::endl;
    add<<<1, 1>>>(N, x, y);
    std::cout << "Exit Into Add" << std::endl;

    std::cout << "Wait for device to synchronize" << std::endl;
    cudaDeviceSynchronize();
    std::cout << "Finish device synchronization" << std::endl;

    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }

    std::cout << "Max Error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);

    return 0;
}