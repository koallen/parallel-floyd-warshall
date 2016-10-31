/*
 * Basic version of Floyd-Warshall
 */
#include <cuda_runtime.h>

#include "Floyd.h"

void Floyd_Warshall(int *matrix, int size)
{
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size, size, 1);

    // run kernel
    for(int k = 0; k < size; ++k)
        run<<<dimGrid, 1>>>(matrixOnGPU, size, k);

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

__global__ void run(int *matrix, int size, int k)
{
    // compute indexes
    int i = blockIdx.y;
    int j = blockIdx.x;

    int i0 = i * size + j;
    int i1 = i * size + k;
    int i2 = k * size + j;

    // read in dependent values
    int i_j_value = matrix[i0];
    int i_k_value = matrix[i1];
    int k_j_value = matrix[i2];

    // calculate shortest path
    if(i_k_value != -1 && k_j_value != -1)
    {
        int sum = i_k_value + k_j_value;
        if (i_j_value == -1 || sum < i_j_value)
            matrix[i0] = sum;
    }
}
