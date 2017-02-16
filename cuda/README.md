# CUDA implementation

This implementation follows the idea presented in

> Katz, Gary J., and Joseph T. Kider Jr. "All-pairs shortest-paths for large graphs on the GPU." In Proceedings of the 23rd ACM SIGGRAPH/EUROGRAPHICS symposium on Graphics hardware, pp. 47-55. Eurographics Association, 2008.

## Running the code

Please change the value in `debug.h`

- If you want to run basic version, set `APSP_VER` to 1
- If you want to run coalesced access optimized version, set `APSP_VER` to 2
- If you want to run shared memory optimized version, set `APSP_VER` to 3
- If you want to run fully optimized version , set `APSP_VER` to 4

Also remember to modify `Makefile` to change the dependency of target `test`

- If you want to run basic version, set `Floyd.o` as dependency
- If you want to run coalesced access optimized version, set `Floyd_coa.o` as dependency
- If you want to run shared memory optimized version, set `Floyd_sm.o` as dependency
- If you want to run fully optimized version, set `Floyd_blk.o` as dependency

**Please run `make clean && make` after you change the above settings each time**

**If you want to profile CUDA kernel, set `PROFILING` in `debug.h`**
