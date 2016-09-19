#include <omp.h>
#include <string.h>
#include <stdlib.h>

#include "Floyd_openMP.c"

void Floyd_Warshall(int[][] matrix, int size){


	int i, j, k;


	for (k = 0; k < size; k++){
		#pragma omp parallel for private(j) schedule(static)
		for (i = 0; i < size; i++){
			for (j = 0; j < size; j++){
				if (matrix[i][k] != -1 && matrix[k][j] != -1){
					int new_path = matrix[i][k] + matrix[k][j];
					if (new_path < matrix[i][j] || matrix[i][j] == -1)
						matrix[i][j] = new_path;
				}
			}
		}
	}	

}