# Comprehensive Documentation for CUDA GEMM Implementation

This document provides a comprehensive overview of the CUDA GEMM project, a collection of General Matrix Multiplication (GEMM) kernels implemented in CUDA C++. The project explores various optimization techniques, from basic naive implementations to highly optimized kernels that leverage NVIDIA's Tensor Cores.

## 1. Project Overview

The primary goal of this project is to explore and implement various GEMM optimization techniques on NVIDIA GPUs. It serves as a practical guide to understanding how to achieve high performance in CUDA programming by progressively optimizing a common and critical operation like GEMM. The performance of each implemented kernel is benchmarked against NVIDIA's highly optimized cuBLAS library.

## 2. Repository Structure

The repository is organized as follows:

- **`src/`**: Contains the source code for all the GEMM kernels.
    - **`basic/`**: Basic, easy-to-understand GEMM implementations.
        - `naive_gemm.cuh`: A simple, unoptimized GEMM kernel.
        - `tiled_gemm.cuh`: A GEMM kernel that uses shared memory to reduce global memory accesses.
        - `optimum_tiled_gemm.cuh`: A more optimized tiled GEMM kernel.
    - **`cublas/`**: A wrapper for the cuBLAS library.
        - `cublas_gemm.cuh`: A wrapper for `cublasSgemm` and `cublasGemmEx`.
    - **`cute/`**: An implementation using the CUTE library.
        - `sgemm_cute_naive.cuh`: A GEMM kernel implemented using the CUTE library.
    - **`mma/`**: A kernel that uses the `mma.sync` PTX instruction.
        - `hgemm_m16n8k16.cuh`: A GEMM kernel that uses Tensor Cores directly via PTX.
    - **`wmma/`**: Kernels that use the `nvcuda::wmma` API for Tensor Core acceleration.
        - `wmma_gemm_naive_fp16.cuh`: A basic WMMA kernel.
        - `hgemm_wmma_m16n16k16.cuh`: A more structured version of the naive WMMA kernel.
        - `hgemm_wmma_m16n16k16_mma4x2.cuh`: Increases the amount of work per block.
        - `hgemm_wmma_mnk16_mma4x2_warp2x4.cuh`: Further increases the work per block.
        - `hgemm_wmma_mnk16_m4x2_dbuf_async.cuh`: The most advanced kernel, using double buffering and asynchronous data transfers.
- **`inc/`**: Header files.
    - `launch.h`: Contains function declarations for launching the kernels.
    - `util.cuh`: Utility functions for benchmarking, verification, and initialization.
- **`benchmark.py`**: A Python script to plot the benchmark results.
- **`CMakeLists.txt`**: The build script for the project.
- **`README.md`**: A brief overview of the project.
- **`docs.md`**: This documentation file.

## 3. Kernels Implemented

### 3.1. Basic Kernels

#### `naive_gemm.cuh`
- **Description**: This is the most straightforward implementation of GEMM. Each thread computes a single element of the output matrix `C`. It involves a simple triple-nested loop structure.
- **Performance**: This kernel is the slowest and is used to demonstrate the baseline performance without any optimizations.

#### `tiled_gemm.cuh`
- **Description**: This kernel introduces the concept of tiling (or blocking). It loads small tiles of the input matrices `A` and `B` into shared memory to reduce global memory access and exploit data reuse. Each thread block computes a tile of the output matrix `C`.
- **Performance**: Tiling significantly improves performance compared to the naive kernel by reducing the number of costly global memory accesses.

#### `optimum_tiled_gemm.cuh`
- **Description**: This kernel is a more optimized version of the tiled GEMM. It further refines the tiling strategy and memory access patterns to improve performance.
- **Performance**: This kernel offers better performance than the basic tiled GEMM through more efficient use of shared memory and thread-level parallelism.

### 3.2. cuBLAS Kernel

#### `cublas_gemm.cuh`
- **Description**: This is not a custom kernel but a wrapper around NVIDIA's highly optimized `cublasSgemm` and `cublasGemmEx` functions. It is used as the gold standard for performance comparison.
- **Performance**: cuBLAS provides a highly optimized GEMM implementation that is difficult to match. It serves as the primary benchmark for our custom kernels.

### 3.3. CUTE Kernel

#### `sgemm_cute_naive.cuh`
- **Description**: This kernel is an implementation of GEMM using NVIDIA's CUTE (CUDA Unbound Tensors and Execution) library. CUTE provides a powerful and flexible way to describe and manipulate tensors, which simplifies the development of complex kernels.
- **Performance**: This kernel demonstrates the use of CUTE for GEMM. While it is a "naive" implementation within the CUTE framework, it can achieve good performance due to CUTE's ability to generate efficient code.

### 3.4. MMA Kernel

#### `hgemm_m16n8k16.cuh`
- **Description**: This kernel utilizes the `mma.sync.aligned.m16n8k16` PTX instruction, which is a direct way to use Tensor Cores for matrix multiplication. It performs a 16x8x16 matrix multiply-accumulate operation.
- **Performance**: By using Tensor Cores, this kernel achieves a significant performance boost over the basic kernels. It's a more direct, lower-level approach compared to the WMMA API.

### 3.5. WMMA Kernels

These kernels use the `nvcuda::wmma` API, which provides a C++ interface for using Tensor Cores. The WMMA kernels are progressively optimized.

#### `wmma_gemm_naive_fp16.cuh`
- **Description**: This is the most basic WMMA kernel. Each warp is responsible for a 16x16 tile of the output matrix. It loads data directly from global memory into fragments and performs the matrix multiplication.
- **Performance**: This kernel is a simple introduction to WMMA and provides a good performance uplift over non-Tensor Core kernels.

#### `hgemm_wmma_m16n16k16.cuh`
- **Description**: A slightly more structured version of the naive WMMA kernel, with a clearer separation of concerns.
- **Performance**: Similar to the naive WMMA kernel.

#### `hgemm_wmma_m16n16k16_mma4x2.cuh`
- **Description**: This kernel increases the amount of work per block by having each block compute a larger tile of the output matrix (64x32). It uses shared memory to stage the data for the WMMA operations, which improves data reuse and reduces global memory traffic.
- **Performance**: The use of shared memory and increased work per block leads to a significant performance improvement.

#### `hgemm_wmma_mnk16_mma4x2_warp2x4.cuh`
- **Description**: This kernel further increases the work per block to compute a 128x128 tile of the output matrix. This is done to improve occupancy and memory bandwidth utilization.
- **Performance**: This kernel shows another significant performance jump due to the larger tile size.

#### `hgemm_wmma_mnk16_m4x2_dbuf_async.cuh`
- **Description**: This is the most advanced kernel in this project. It builds upon the previous kernel by introducing **double buffering** and **asynchronous data transfers** (`cp.async`).
    -   **Double Buffering**: Two sets of shared memory buffers are used for `A` and `B` tiles. While the GPU is performing computations on one set of buffers, the next set of tiles is being fetched from global memory in the background.
    -   **Asynchronous Data Transfers**: The `cp.async` PTX instruction allows for data to be copied from global to shared memory without blocking the SMs, effectively hiding the memory latency.
- **Performance**: This kernel achieves the highest performance, coming very close to the performance of cuBLAS. The combination of Tensor Cores, shared memory, large tile sizes, double buffering, and asynchronous data transfers allows for near-optimal utilization of the GPU's resources.

## 4. Building and Running

### Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 8.0+ for Tensor Cores)
- CUDA Toolkit
- CMake (version 3.18 or higher)

### Building the Project

1.  Create a `build` directory: `mkdir build`
2.  Navigate to the `build` directory: `cd build`
3.  Run CMake: `cmake ..`
4.  Compile the project: `make`

### Running the Benchmark

After building the project, you can run the benchmark from the `build` directory:

```bash
./benchmark
```

This will run all the implemented kernels and print the performance results to the console.

## 5. Benchmark Results

The performance of the implemented kernels was benchmarked against NVIDIA's cuBLAS library. The benchmark was run on an NVIDIA GPU with Tensor Cores.

![Benchmark Plot](assets/benchmark_plot.png)

The plot above shows the TFLOPS achieved by each kernel for different matrix sizes. As you can see, the `hgemm_wmma_mnk16_m4x2_dbuf_async` kernel achieves performance that is very close to cuBLAS, demonstrating the effectiveness of the optimization techniques used.