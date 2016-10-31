# CUDA implementation

- If you want to run basic version, set `APSP_VER` to 1
- If you want to run coalesced access optimized version, set `APSP_VER` to 2
- If you want to run coalesced access optimized version, set `APSP_VER` to 3
- If you want to run fully optimized version ,set `APSP_VER` to 4

Also remember to modify `Makefile` to change the dependency of target `test`.
