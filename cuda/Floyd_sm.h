void Floyd_Warshall(int *matrix, int size);
__global__ void run(int *matrix, int size, int k);

#define TILE_WIDTH 256
