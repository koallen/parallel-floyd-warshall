/*
 * Floyd-Warshall with only coalesced memory optimization
 */
#include <cuda_runtime.h>

#include "Floyd_coa.h"

void Floyd_Warshall(int *matrix, int size)
{
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size / TILE_WIDTH, size / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // run kernel
    for(int k = 0; k < size; ++k)
        run<<<dimGrid, dimBlock>>>(matrixOnGPU, size, k);

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

__global__ void run(int *matrix, int size, int k)
{
    // compute the indexes
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int i0 = i * size + j;
    int i1 = i * size + k;
    int i2 = k * size + j;

    // read in data
    int i_j_value = matrix[i0];
    int i_k_value = matrix[i1];
    int k_j_value = matrix[i2];

    // Floyd-Warshall
    if(i_k_value != -1 && k_j_value != -1)
    {
        int sum = i_k_value + k_j_value;
        if (i_j_value == -1 || sum < i_j_value)
            matrix[i0] = sum;
    }
}
