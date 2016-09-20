#include <omp.h>
#include <string.h>
#include <stdlib.h>

#include "Floyd_openMP.h"

void Floyd_Warshall(int* matrix, int size){

	int i, j, k;
    int *row_k = (int*)malloc(sizeof(int)*size);

	for (k = 0; k < size; k++){
        memcpy(row_k, matrix + (k * size), sizeof(int)*size);
        #pragma omp parallel for private(j) schedule(static)
        for (i = 0; i < size; ++i) {
            for (j = 0; j < size; ++j) {
                if (matrix[i * size + k] != -1 && row_k[j] != -1){
                    int new_path = matrix[i * size + k] + row_k[j];
                    if (new_path < matrix[i * size + j] || matrix[i * size + j] == -1)
                        matrix[i*size+j] = new_path;
                }
            }
        }
    }
}
