#include <mpi.h>
#include <string.h>
#include <stdlib.h>

#include "Variables.h"
#include "Floyd.h"

int process_count, process_rank;

void PL_APSP(int *matrix, int size, int *result)
{
    int sum, start, end, portion = size / process_count;

    // to store row k
    int *row_k = (int*)malloc(sizeof(int)*size);

    for (int k = 0; k < size; ++k)
    {
        if (process_rank == k/portion)
        {
            memcpy(row_k, matrix+(k%portion)*size, sizeof(int)*size);
        }
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

    MPI_Gather(matrix, portion * size, MPI_INT, result, portion * size, MPI_INT, 0, MPI_COMM_WORLD);
}
