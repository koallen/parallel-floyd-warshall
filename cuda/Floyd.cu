#include "Floyd.h"

#define TILE_WIDTH 16

void Floyd_Warshall(int *matrix, int size)
{
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int) * size * size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size / TILE_WIDTH, size / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // run kernel
    for(int k = 0; k < size; k ++)
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
        if(matrix[i1] != -1 && matrix[i2] != -1)
        {
            int sum =  (matrix[i1] + matrix[i2]);
            if (matrix[i0] == -1 || sum < matrix[i0])
                matrix[i0] = sum;
        }
    }
}
