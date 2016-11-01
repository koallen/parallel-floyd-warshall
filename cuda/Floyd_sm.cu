/*
 * Floyd-Warshall with shared memory optimization
 */
#include <cuda_runtime.h>

#include "Floyd_sm.h"

void Floyd_Warshall(int *matrix, int size)
{
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size, size / TILE_WIDTH, 1);
    dim3 dimBlock(1, TILE_WIDTH, 1);

    // run kernel
    for(int k = 0; k < size; ++k)
        run<<<dimGrid, dimBlock>>>(matrixOnGPU, size, k);

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

__global__ void run(int *matrix, int size, int k)
{
    // compute indexes
    int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int j = blockIdx.x;

    int i0 = i * size + j;
    int i1 = i * size + k;
    int i2 = k * size + j;

    // read in dependent values
    int i_j_value = matrix[i0];
    int i_k_value = matrix[i1];
    __shared__ int k_j_value;
    if (threadIdx.y == 0)
        k_j_value = matrix[i2];
    __syncthreads();

    // calculate shortest path
    if(i_k_value != -1 && k_j_value != -1)
    {
        int sum = i_k_value + k_j_value;
        if (i_j_value == -1 || sum < i_j_value)
            matrix[i0] = sum;
    }
}
