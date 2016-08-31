#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#include "MatUtil.h"
#include "Variables.h"
#include "Floyd.h"

// global variable
int process_count, process_rank;

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: test {N}\n");
		exit(-1);
	}

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    size_t N = atoi(argv[1]);
    int *mat, *ref;

    struct timeval tv1, tv2;

    // master process will compute the sequential version
    if (process_rank == 0)
    {
        //generate a random matrix.
        mat = (int*)malloc(sizeof(int)*N*N);
        GenMatrix(mat, N);

        //compute the reference result.
        ref = (int*)malloc(sizeof(int)*N*N);
        memcpy(ref, mat, sizeof(int)*N*N);
        gettimeofday(&tv1, NULL);
        ST_APSP(ref, N);
        gettimeofday(&tv2, NULL);
        printf("Elasped time = %ld usecs\n", (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // transfer generated matrix
    int *result = (int*)malloc(sizeof(int)*N*N);
    int *matrix = (int*)malloc(sizeof(int)*N*(N/process_count));
    if (process_rank == 0)
        gettimeofday(&tv1, NULL);
    MPI_Scatter(mat, N*(N/process_count), MPI_INT, matrix, N*(N/process_count), MPI_INT, 0, MPI_COMM_WORLD);

    // all processes will compute the parallel version
    PL_APSP(matrix, N, result);
    if (process_rank == 0)
    {
        gettimeofday(&tv2, NULL);
        printf("Elasped time = %ld usecs\n", (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // master process will verify the result
    if (process_rank == 0)
    {
        //compare your result with reference result
        if(CmpArray(result, ref, N*N))
            printf("Your result is correct.\n");
        else
            printf("Your result is wrong.\n");
    }

    MPI_Finalize();
}
