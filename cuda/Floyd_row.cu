#include <cuda_runtime.h>

#include "Floyd_row.h"

void Floyd_Warshall(int *matrix, int size)
{
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size / TILE_WIDTH, size, 1);
    dim3 dimBlock(TILE_WIDTH, 1, 1);

    // run kernel
    for(int k = 0; k < size; ++k)
        run<<<dimGrid, dimBlock>>>(matrixOnGPU, size, k);

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

__global__ void run(int *matrix, int size, int k)
{
    // get thread index
    int i = blockIdx.y;
    int j = blockIdx.x * TILE_WIDTH + threadIdx.x;

    int i0 = i * size + j;
    int i1 = i * size + k;
    int i2 = k * size + j;

    // shared memory
    __shared__ int i_k;
    __shared__ int k_j[TILE_WIDTH];
    k_j[threadIdx.x] = matrix[i2]; // read in all the k_j
    if (threadIdx.x == 0)
        i_k = matrix[i1]; // read in i_k

    int i0_value = matrix[i0]; // read in i_j

    __syncthreads(); // sync before compute

    int k_j_value = k_j[threadIdx.x];
    if(i_k != -1 && k_j_value != -1)
    {
        int sum = i_k + k_j_value;
        if (i0_value == -1 || sum < i0_value)
            matrix[i0] = sum;
    }
}
