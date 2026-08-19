// RTX 5090 / sm_120 unity translation unit. Keeping CUDAHash and the search
// kernel together lets nvcc optimize the fixed-size HASH160 hot path without
// changing the original multi-TU baseline build.
#include "CUDAHash.cu"
#include "CUDACyclone.cu"
