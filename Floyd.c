#include <mpi.h>
#include <string.h>
#include <stdlib.h>

#include "Floyd.h"

extern int process_count, process_rank;

void Floyd_Warshall(int *mat, int N, int *matrix, int *result)
{
    int portion = N / process_count;
    // data distribution
    MPI_Scatter(mat, N * portion, MPI_INT, matrix, N * portion, MPI_INT, 0, MPI_COMM_WORLD);
    PL_APSP(matrix, N, portion);
    // result collection
    MPI_Gather(matrix, N * portion, MPI_INT, result, N * portion, MPI_INT, 0, MPI_COMM_WORLD);
}

void PL_APSP(int *matrix, int size, int portion)
{
    int sum, start, end, owner_of_k, offset_to_k;
    int k, i, j;
    int *row_k = (int*)malloc(sizeof(int) * size); // to store row k

    for (k = 0; k < size; ++k)
    {
        owner_of_k = k / portion;
        offset_to_k = (k % portion) * size;

        // broadcast kth row
        if (process_rank == owner_of_k)
            memcpy(row_k, matrix + offset_to_k, sizeof(int) * size);
        MPI_Bcast(row_k, size, MPI_INT, owner_of_k, MPI_COMM_WORLD);

        for (i = 0; i < portion; ++i)
        {
            for (j = 0; j < size; ++j)
            {
                if (row_k[j] != -1 && matrix[i * size + k] != -1)
                {
                    sum = matrix[i * size + k] + row_k[j];
                    if (matrix[i * size + j] > sum || matrix[i * size + j] == -1)
                        matrix[i * size + j] = sum;
                }
            }
        }
    }
}
