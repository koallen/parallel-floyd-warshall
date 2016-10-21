#include <iostream>
#include <cstdlib>
#include <sys/time.h>

#include <cuda_runtime.h>

#include "MatUtil.h"
#include "Floyd.h"

using namespace std;

int main(int argc, char **argv)
{
    struct timeval tv1, tv2;

    if(argc != 2)
    {
        cout << "Usage: test {N}" << endl;
        exit(-1);
    }

    //generate a random matrix.
    size_t N = atoi(argv[1]);
    int *mat = (int*)malloc(sizeof(int)*N*N);
    GenMatrix(mat, N);

    //compute the reference result.
    int *ref = (int*)malloc(sizeof(int)*N*N);
    memcpy(ref, mat, sizeof(int)*N*N);
    gettimeofday(&tv1, NULL);
    ST_APSP(ref, N);
    gettimeofday(&tv2, NULL);
    long sequentialtime = (tv2.tv_sec - tv1.tv_sec)*1000000 + tv2.tv_usec - tv1.tv_usec;
    cout << "Elapsed time (sequential) = " << sequentialtime << " usecs" << endl;

    //compute your results
    int *result = (int*)malloc(sizeof(int)*N*N);
    memcpy(result, mat, sizeof(int)*N*N);
    //replace by parallel algorithm
    gettimeofday(&tv1, NULL);
    //Floyd_Warshall(result, N);
    hello<<<1, 32>>>();
    gettimeofday(&tv2, NULL);
    long paralleltime = (tv2.tv_sec - tv1.tv_sec)*1000000 + tv2.tv_usec - tv1.tv_usec;
    cout << "Elapsed time (parallel) = " << paralleltime << " usecs" << endl;

    cout << "Speed up = " << (double)sequentialtime/paralleltime << endl;
    //compare your result with reference result
    if(CmpArray(result, ref, N*N))
        cout << "Your result is correct." << endl;
    else
        cout << "Your result is wrong." << endl;
}
