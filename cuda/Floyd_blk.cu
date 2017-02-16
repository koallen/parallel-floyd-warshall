#include <cuda_runtime.h>

#include "Floyd_blk.cuh"

void Floyd_Warshall(int *matrix, int size)
{
    int stages = size / TILE_WIDTH;

    // allocate memory
    int *matrixOnGPU;
    cudaMalloc(&matrixOnGPU, sizeof(int) * size * size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int) * size * size, cudaMemcpyHostToDevice);

    // dimensions
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 phase1Grid(1, 1, 1);
    dim3 phase2Grid(stages, 2, 1);
    dim3 phase3Grid(stages, stages, 1);

    // run kernel
    for(int k = 0; k < stages; ++k)
    {
		int base = TILE_WIDTH * k;
        phase1<<<phase1Grid, blockSize>>>(matrixOnGPU, size, base);
        phase2<<<phase2Grid, blockSize>>>(matrixOnGPU, size, k, base);
        phase3<<<phase3Grid, blockSize>>>(matrixOnGPU, size, k, base);
    }

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int) * size * size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

/*
 * This kernel computes the first phase (self-dependent block)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param size   The width of the matrix
 * @param base   The base index for a block
 */
__global__ void phase1(int *matrix, int size, int base)
{
    // computes the index for a thread
    int index = (base + threadIdx.y) * size + (base + threadIdx.x);

    // loads data from global memory to shared memory
    __shared__ int subMatrix[TILE_WIDTH][TILE_WIDTH];
    subMatrix[threadIdx.y][threadIdx.x] = matrix[index];
    __syncthreads();

    // run Floyd-Warshall
    int sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        sum = subMatrix[threadIdx.y][k] + subMatrix[k][threadIdx.x];
        if (sum < subMatrix[threadIdx.y][threadIdx.x])
            subMatrix[threadIdx.y][threadIdx.x] = sum;
    }

    // write back to global memory
    matrix[index] = subMatrix[threadIdx.y][threadIdx.x];
}

/*
 * This kernel computes the second phase (singly-dependent blocks)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param size   The width of the matrix
 * @param stage  The current stage of the algorithm
 * @param base   The base index for a block
 */
__global__ void phase2(int *matrix, int size, int stage, int base)
{
    // computes the index for a thread
    if (blockIdx.x == stage) return;

    int i, j, i_prim, j_prim;
    i_prim = base + threadIdx.y;
    j_prim = base + threadIdx.x;
    if (blockIdx.y) // load for column
    {
        i = TILE_WIDTH * blockIdx.x + threadIdx.y;
        j = j_prim;
    } else { // load for row
        j = TILE_WIDTH * blockIdx.x + threadIdx.x;
        i = i_prim;
    }
    int index = i * size + j;
    int index_prim = i_prim * size + j_prim;

    // loads data from global memory to shared memory
    __shared__ int ownMatrix[TILE_WIDTH][TILE_WIDTH];
    __shared__ int primaryMatrix[TILE_WIDTH][TILE_WIDTH];
    ownMatrix[threadIdx.y][threadIdx.x] = matrix[index];
    primaryMatrix[threadIdx.y][threadIdx.x] = matrix[index_prim];
    __syncthreads();

    // run Floyd Warshall
    int sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        sum = ownMatrix[threadIdx.y][k] + primaryMatrix[k][threadIdx.x];
        if (sum < ownMatrix[threadIdx.y][threadIdx.x])
            ownMatrix[threadIdx.y][threadIdx.x] = sum;
    }

    // write back to global memory
    matrix[index] = ownMatrix[threadIdx.y][threadIdx.x];
}


/*
 * This kernel computes the third phase (doubly-dependent blocks)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param size   The width of the matrix
 * @param stage  The current stage of the algorithm
 * @param base   The base index for a block
 */
__global__ void phase3(int *matrix, int size, int stage, int base)
{
    // computes the index for a thread
    if (blockIdx.x == stage || blockIdx.y == stage) return;

    int i, j, j_col, i_row;
    i = TILE_WIDTH * blockIdx.y + threadIdx.y;
    j = TILE_WIDTH * blockIdx.x + threadIdx.x;
    i_row = base + threadIdx.y;
    j_col = base + threadIdx.x;
    int index, index_row, index_col;
    index = i * size + j;
    index_row = i_row * size + j;
    index_col = i * size + j_col;

    // loads data from global memory into shared memory
    __shared__ int rowMatrix[TILE_WIDTH][TILE_WIDTH];
    __shared__ int colMatrix[TILE_WIDTH][TILE_WIDTH];
    int i_j = matrix[index];
    rowMatrix[threadIdx.y][threadIdx.x] = matrix[index_row];
    colMatrix[threadIdx.y][threadIdx.x] = matrix[index_col];
    __syncthreads();

    // run Floyd Warshall
    int sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        sum = colMatrix[threadIdx.y][k] + rowMatrix[k][threadIdx.x];
        if (sum < i_j)
		    i_j = sum;
    }

    // write back to global memory
    matrix[index] = i_j;
}
