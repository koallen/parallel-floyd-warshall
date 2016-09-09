#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#include "MatUtil.h"
#include "Floyd.h"

// global variables
int process_count, process_rank;

int main(int argc, char **argv)
{
    size_t N;
    int *mat, *ref, *result, *matrix;
    struct timeval tv1, tv2;

    if(argc != 2)
    {
        printf("Usage: test {N}\n");
        exit(-1);
    }

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    N = atoi(argv[1]);
    matrix = (int*)malloc(sizeof(int) * N * (N / process_count));

    // master process will compute the sequential version
    if (process_rank == 0)
    {
        // generate a random matrix
        mat = (int*)malloc(sizeof(int)*N*N);
        GenMatrix(mat, N);

        // compute the reference result
        ref = (int*)malloc(sizeof(int)*N*N);
        memcpy(ref, mat, sizeof(int)*N*N);
        gettimeofday(&tv1, NULL);
        ST_APSP(ref, N);
        gettimeofday(&tv2, NULL);
        printf("Elasped time (sequential) = %ld usecs\n", (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);

        // allocate space for parallel implementation result
        result = (int*)malloc(sizeof(int) * N * N);
    }

    // make sure all processes are in sync before we start
    MPI_Barrier(MPI_COMM_WORLD);

    if (process_rank == 0)
        gettimeofday(&tv1, NULL);
    // run the parallel implementation
    Floyd_Warshall(mat, N, matrix, result);
    if (process_rank == 0)
    {
        gettimeofday(&tv2, NULL);
        printf("Elasped time (parallel)   = %ld usecs\n", (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec);
    }

    // master process will verify the result
    if (process_rank == 0)
    {
        // compare your result with reference result
        if(CmpArray(result, ref, N*N))
            printf("Your result is correct.\n");
        else
            printf("Your result is wrong.\n");
    }

    MPI_Finalize();
}
