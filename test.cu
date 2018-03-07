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
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
	y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

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