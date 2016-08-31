#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "MatUtil.h"

void PL_APSP(int *matrix, int size);

int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: test {N}\n");
		exit(-1);
	}

	//generate a random matrix.
	size_t N = atoi(argv[1]);
	int *mat = (int*)malloc(sizeof(int)*N*N);
	GenMatrix(mat, N);

	//compute the reference result.
	int *ref = (int*)malloc(sizeof(int)*N*N);
	memcpy(ref, mat, sizeof(int)*N*N);
	ST_APSP(ref, N);

	//compute your results
	int *result = (int*)malloc(sizeof(int)*N*N);
	memcpy(result, mat, sizeof(int)*N*N);
	//replace by parallel algorithm
    MPI_Init(&argc, &argv);
    PL_APSP(result, N);
    MPI_Finalize();

	//compare your result with reference result
	if(CmpArray(result, ref, N*N))
		printf("Your result is correct.\n");
	else
		printf("Your result is wrong.\n");
}

void PL_APSP(int *matrix, int size)
{
    int processor_count, processor_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &processor_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &processor_rank);
    printf("rank: %d, total: %d\n", processor_rank, processor_count);
}
