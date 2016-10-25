#include "Floyd.h"

#define TILE_WIDTH 16

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
    // get thread index
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size && j < size)
    {
        int i0 = i * size + j;
        int i1 = i * size + k;
        int i2 = k * size + j;
        int i0_value = matrix[i0];
        int i1_value = matrix[i1];
        int i2_value = matrix[i2];
        if(i1_value != -1 && i2_value != -1)
        {
            int sum = i1_value + i2_value;
            if (i0_value == -1 || sum < i0_value)
                matrix[i0] = sum;
        }
    }
}
