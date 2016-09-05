#include <mpi.h>
#include <string.h>
#include <stdlib.h>

#include "Variables.h"
#include "Floyd.h"

// global variables
int process_count, process_rank;

void Floyd_Warshall(int *mat, int N, int process_count, int *matrix, int *result)
{
    // data distribution
    MPI_Scatter(mat, N * (N / process_count), MPI_INT, matrix, N * (N / process_count), MPI_INT, 0, MPI_COMM_WORLD);
    PL_APSP(matrix, N);
    // result collection
    MPI_Gather(matrix, N * (N / process_count), MPI_INT, result, N * (N / process_count), MPI_INT, 0, MPI_COMM_WORLD);
}

void PL_APSP(int *matrix, int size)
{
    int sum, start, end, portion = size / process_count;
    int *row_k; // to store row k

    row_k = (int*)malloc(sizeof(int)*size);

    for (int k = 0; k < size; ++k)
    {
        // broadcast kth row
        if (process_rank == k/portion)
            memcpy(row_k, matrix+(k%portion)*size, sizeof(int)*size);
        MPI_Bcast(row_k, size, MPI_INT, k/portion, MPI_COMM_WORLD);

        for (int i = 0; i < portion; ++i)
        {
            for (int j = 0; j < size; ++j)
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
