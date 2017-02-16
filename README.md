# Parallel Floyd-Warshall

Parallelized implementation of the Floyd-Warshall algorithm.

## Implementations

The Floyd-Warshall algorithm has been implemented with MPI, OpenMP, and CUDA.

## Speed up

Almost linear speedup for MPI and OpenMP (tested with < 10 processors). 600+ speed up for CUDA with a graph of size 4096 ^ 2 (tested with NVIDIA GTX 770).
