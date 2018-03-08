#include <cuda_runtime.h>
const int N = 1<<20;

__global__
void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    float a[N][N], b[N][N], c[N][N];
    int *c_a, *c_b, *c_c;
    size_t pitch_a, pitch_b, pitch_c;

    // how to allocate memory in 2D??
    cudaMallocPitch(&c_a, &pitch_a, N * sizeof(float), N);
    cudaMallocPitch(&c_b, &pitch_b, N * sizeof(float), N);
    cudaMallocPitch(&c_c, &pitch_c, N * sizeof(float), N);
    cudaMemcpy2D(c_a, pitch_a, a, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);
    cudaMemcpy2D(c_b, pitch_b, b, N * sizeof(float), N * sizeof(float), N, cudaMemcpyHostToDevice);

    int numBlocks = 1;
    int threadSize = 64;
    dim3 threadsPerBlock(threadSize, threadSize);
    MatAdd<<<numBlocks, threadsPerBlock>>>(a, b, c);

    cudaDeviceSynchronize();

    cudaMemcpy2D(c_c, pitch_c, c, N * sizeof(float), N * sizeof(float), N, cudaMemcpyDeviceToHost);

    cudaFree(c_a);
    cudaFree(c_b);
    cudaFree(c_c);

    return 0;
}