#include <cuda_runtime.h>

#include "Floyd_blk.h"

void Floyd_Warshall(int *matrix, int size)
{
    /*cout << "started running blocked Floyd-Warshall" << endl;*/
    int stages = size / TILE_WIDTH;

    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int)*size*size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int)*size*size, cudaMemcpyHostToDevice);

    // dimension
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 phase1Grid(1, 1, 1);
    dim3 phase2Grid(stages, 2, 1);
    dim3 phase3Grid(stages, stages, 1);

    // run kernel
    for(int k = 0; k < stages; ++k)
    {
        phase1<<<phase1Grid, blockSize>>>(matrixOnGPU, size, k);
        phase2<<<phase2Grid, blockSize>>>(matrixOnGPU, size, k);
        phase3<<<phase3Grid, blockSize>>>(matrixOnGPU, size, k);
    }

    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

/*
 * This kernel computes the first phase (self-dependent block)
 *
 * @param matrix A pointer to the adjacency matrix
 * @param size   The width of the matrix
 * @param stage  The current stage of the algorithm
 */
__global__ void phase1(int *matrix, int size, int stage)
{
    // computes the index for a thread
    int base = TILE_WIDTH * stage;
    int i = base + threadIdx.y;
    int j = base + threadIdx.x;
    int index = i * size + j;

    // loads data from global memory to shared memory
    __shared__ int subMatrix[TILE_WIDTH][TILE_WIDTH];
    subMatrix[threadIdx.y][threadIdx.x] = matrix[index];
    __syncthreads();

    // run Floyd-Warshall
    int i_k, k_j, i_j, sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        i_j = subMatrix[threadIdx.y][threadIdx.x];
        i_k = subMatrix[threadIdx.y][k];
        k_j = subMatrix[k][threadIdx.x];
        if (i_k != -1 && k_j != -1)
        {
            sum = i_k + k_j;
            if (i_j == -1 || sum < i_j)
                subMatrix[threadIdx.y][threadIdx.x] = sum;
        }
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
 */
__global__ void phase2(int *matrix, int size, int stage)
{
    // computes the index for a thread
    if (blockIdx.x == stage)
        return;
    int base = TILE_WIDTH * stage;
    /*int skip_center = min((blockIdx.x+1)/(stage+1), 1);*/
    int col = blockIdx.y;
    int i, j, i_prim, j_prim;
    i_prim = base + threadIdx.y;
    j_prim = base + threadIdx.x;
    if (col) // load for column
    {
        /*i = TILE_WIDTH * (blockIdx.x + skip_center) + threadIdx.y;*/
        i = TILE_WIDTH * blockIdx.x + threadIdx.y;
        j = j_prim;
    } else { // load for row
        j = TILE_WIDTH * blockIdx.x + threadIdx.x;
        i = i_prim;
        /*j = TILE_WIDTH * (blockIdx.x + skip_center) + threadIdx.x;*/
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
    int i_k, k_j, i_j, sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        i_j = ownMatrix[threadIdx.y][threadIdx.x];
        i_k = ownMatrix[threadIdx.y][k];
        k_j = primaryMatrix[k][threadIdx.x];
        if (i_k != -1 && k_j != -1)
        {
            sum = i_k + k_j;
            if (i_j == -1 || sum < i_j)
                ownMatrix[threadIdx.y][threadIdx.x] = sum;
        }
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
 */
__global__ void phase3(int *matrix, int size, int stage)
{
    // computes the index for a thread
    if (blockIdx.x == stage || blockIdx.y == stage)
        return;
    /*int skip_center_x = min((blockIdx.x+1)/(stage+1), 1);*/
    /*int skip_center_y = min((blockIdx.y+1)/(stage+1), 1);*/
    int base = TILE_WIDTH * stage;
    int i, j, i_col, j_col, i_row, j_row;
    /*i = TILE_WIDTH * (blockIdx.y + skip_center_y) + threadIdx.y;*/
    /*j = TILE_WIDTH * (blockIdx.x + skip_center_x) + threadIdx.x;*/
    i = TILE_WIDTH * blockIdx.y + threadIdx.y;
    j = TILE_WIDTH * blockIdx.x + threadIdx.x;
    i_row = base + threadIdx.y;
    j_row = j;
    i_col = i;
    j_col = base + threadIdx.x;
    int index, index_row, index_col;
    index = i * size + j;
    index_row = i_row * size + j_row;
    index_col = i_col * size + j_col;

    // loads data from global memory into shared memory
    __shared__ int rowMatrix[TILE_WIDTH][TILE_WIDTH];
    __shared__ int colMatrix[TILE_WIDTH][TILE_WIDTH];
    int i_j = matrix[index];
    rowMatrix[threadIdx.y][threadIdx.x] = matrix[index_row];
    colMatrix[threadIdx.y][threadIdx.x] = matrix[index_col];
    __syncthreads();

    // run Floyd Warshall
    int i_k, k_j, sum;
    for (int k = 0; k < TILE_WIDTH; ++k)
    {
        i_k = colMatrix[threadIdx.y][k];
        k_j = rowMatrix[k][threadIdx.x];
        if (i_k != -1 && k_j != -1)
        {
            sum = i_k + k_j;
            if (i_j == -1 || sum < i_j)
                i_j = sum;
        }
    }

    // write back to global memory
    matrix[index] = i_j;
}
