void Floyd_Warshall(int *matrix, int size);
__global__ void phase1(int *matrix, int size, int base);
__global__ void phase2(int *matrix, int size, int stage, int base);
__global__ void phase3(int *matrix, int size, int stage, int base);

#define TILE_WIDTH 32
