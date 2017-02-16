#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include "MatUtil.h"
#include "debug.h"

#if (APSP_VER == 1)
    #include "Floyd.h"
#elif (APSP_VER == 2)
    #include "Floyd_coa.h"
#elif (APSP_VER == 3)
    #include "Floyd_sm.h"
#else
    #include "Floyd_blk.cuh"
#endif

#define duration(tv1, tv2) (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec

using namespace std;

int main(int argc, char **argv)
{
    struct timeval tv1, tv2;

    if (argc != 2)
    {
        cout << "Usage: test {N}" << endl;
        exit(-1);
    }

	/*
	 * Input matrix generation
	 */
    size_t N = atoi(argv[1]);
	int matrixElementCount = N * N;
    int *mat = (int*)malloc(sizeof(int) * matrixElementCount);
    GenMatrix(mat, N);

	/*
	 * Sequential Floyd Warshall
	 */
#ifndef PROFILING
    int *ref = (int*)malloc(sizeof(int) * matrixElementCount);
    memcpy(ref, mat, sizeof(int) * matrixElementCount);
    gettimeofday(&tv1, NULL);
    ST_APSP(ref, N);
    gettimeofday(&tv2, NULL);
    long sequentialtime = duration(tv1, tv2);
    cout << "Elapsed time (sequential) = " << sequentialtime << " usecs" << endl;
#endif

    /*
	 * Parallel Floyd Warshall
	 */
    int *result = (int*)malloc(sizeof(int) * matrixElementCount);
    memcpy(result, mat, sizeof(int) * matrixElementCount);
    gettimeofday(&tv1, NULL);
    Floyd_Warshall(result, N);
    gettimeofday(&tv2, NULL);
    long paralleltime = duration(tv1, tv2);
    cout << "Elapsed time (parallel) = " << paralleltime << " usecs" << endl;
#ifdef LOOP
	for (int i = 0; i < LOOP - 1; ++i)
	{
		memcpy(result, mat, sizeof(int) * matrixElementCount);
		gettimeofday(&tv1, NULL);
		Floyd_Warshall(result, N);
		gettimeofday(&tv2, NULL);
		long paralleltime = duration(tv1, tv2);
		cout << "Elapsed time (parallel) = " << paralleltime << " usecs" << endl;
	}
#endif

	/*
	 * Speed up calculation
	 */
#ifndef PROFILING
    cout << "Speed up = " << (double)sequentialtime/paralleltime << endl;
    //compare your result with reference result
    if(CmpArray(result, ref, N*N))
        cout << "Your result is correct." << endl;
    else
        cout << "Your result is wrong." << endl;
#endif
}
