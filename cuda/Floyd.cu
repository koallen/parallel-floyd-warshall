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

    int i0 = i * size + j;
    int i1 = i * size + k;
    int i2 = k * size + j;

    // shared memory
    __shared__ int i_k[TILE_WIDTH];
    __shared__ int k_j[TILE_WIDTH];

    if (i < size && j < size)
    {
        i_k[threadIdx.y] = matrix[i1];
        k_j[threadIdx.x] = matrix[i2];
        int i0_value = matrix[i0];

        __syncthreads(); // sync before compute

        int i_k_value = i_k[threadIdx.y];
        int k_j_value = k_j[threadIdx.x];
        if(i_k_value != -1 && k_j_value != -1)
        {
            int sum = i_k_value + k_j_value;
            if (i0_value == -1 || sum < i0_value)
                matrix[i0] = sum;
        }
    }
}
