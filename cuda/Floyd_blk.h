void Floyd_Warshall(int *matrix, int size);
__global__ void phase1(int *matrix, int size, int k);
__global__ void phase2(int *matrix, int size, int k);
__global__ void phase3(int *matrix, int size, int k);

#define TILE_WIDTH 32
