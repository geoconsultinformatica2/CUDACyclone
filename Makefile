TARGET      := CUDACyclone
RTX5090_TARGET := CUDACyclone-5090
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

OPT_BATCH      ?= 512
OPT_TPB        ?= 128
OPT_MIN_BLOCKS ?= 3
OPT_MAXRREGCOUNT ?=
OPT_LTO        ?= 1
OPT_FAST_MATH  ?= 1
ifeq ($(OPT_FAST_MATH),1)
RTX5090_FAST_MATH := -use_fast_math
endif
RTX5090_CPPFLAGS := -DRTX5090_OPT=1 -DRTX5090_BATCH=$(OPT_BATCH) -DMAX_BATCH_SIZE=$(OPT_BATCH) -DRTX5090_TPB=$(OPT_TPB) -DRTX5090_MIN_BLOCKS=$(OPT_MIN_BLOCKS)
RTX5090_GENCODE  := -gencode arch=compute_120,code=sm_120
RTX5090_NVCC_FLAGS := -O3 $(RTX5090_FAST_MATH) --ptxas-options=-O3,-v $(RTX5090_GENCODE)
RTX5090_LTO_COMPILE_FLAGS := -O3 $(RTX5090_FAST_MATH) -dc -gencode arch=compute_120,code=lto_120
RTX5090_LTO_LINK_FLAGS := -O3 $(RTX5090_FAST_MATH) -dlto -arch=sm_120 --ptxas-options=-O3,-v
ifneq ($(strip $(OPT_MAXRREGCOUNT)),)
RTX5090_NVCC_FLAGS += --maxrregcount=$(OPT_MAXRREGCOUNT)
RTX5090_LTO_LINK_FLAGS += --maxrregcount=$(OPT_MAXRREGCOUNT)
endif

LDFLAGS    := -lcudadevrt -cudart=static

.PHONY: all rtx5090 clean

all: $(TARGET)

rtx5090: $(RTX5090_TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

$(RTX5090_TARGET): CUDACyclone5090.cu CUDACyclone.cu CUDAHash.cu CUDAHash.cuh CUDAMath.h CUDAUtils.h CUDAStructures.h
ifeq ($(OPT_LTO),1)
	$(CC) $(RTX5090_LTO_COMPILE_FLAGS) $(CXXFLAGS) $(RTX5090_CPPFLAGS) CUDACyclone5090.cu -o CUDACyclone5090.lto.o
	$(CC) $(RTX5090_LTO_LINK_FLAGS) $(CXXFLAGS) CUDACyclone5090.lto.o -o $@ $(LDFLAGS)
else
	$(CC) $(RTX5090_NVCC_FLAGS) $(CXXFLAGS) $(RTX5090_CPPFLAGS) CUDACyclone5090.cu -o $@ $(LDFLAGS)
endif

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(RTX5090_TARGET) CUDACyclone-p71 $(OBJ) CUDACyclone.p71.o CUDAHash.p71.o CUDACyclone5090.lto.o

