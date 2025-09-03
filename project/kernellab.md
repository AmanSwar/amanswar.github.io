**KernelLab: Technical Documentation**

## 1. Introduction

KernelLab is a library of high-performance GPU kernels written in CUDA and Triton. It serves as a practical guide and reference for GPU programming, demonstrating various optimization strategies for common computational workloads in deep learning and scientific computing.

The project is structured to provide a clear progression from simple, "naïve" implementations to highly optimized versions that leverage the full capabilities of modern GPU architectures. Each kernel is self-contained and comes with its own set of benchmarks, allowing for a clear understanding of the performance impact of each optimization.

## 2. Core Concepts in GPU Optimization

The optimizations implemented in KernelLab are based on the following core concepts of GPU programming:

- **Memory Hierarchy:** GPUs have a complex memory hierarchy, consisting of global memory, shared memory, and registers. Efficiently managing data movement between these memory spaces is crucial for performance.
- **Parallelism:** GPUs are massively parallel processors. Structuring code to expose as much parallelism as possible is key to achieving high throughput.
- **Memory Coalescing:** Global memory accesses are most efficient when they are "coalesced," meaning that threads in a warp access contiguous memory locations.
- **Shared Memory:** Shared memory is a small, fast, on-chip memory that can be used to cache frequently accessed data, reducing reliance on slower global memory.
- **Warp-Level Primitives:** A warp is a group of 32 threads that execute in lockstep. CUDA provides a set of "warp-level primitives" (e.g., `__shfl_down_sync`) that allow for efficient communication and data sharing between threads in a warp.
- **Tensor Cores:** Modern NVIDIA GPUs include specialized hardware units called Tensor Cores that are designed to accelerate matrix multiplication and other deep learning operations.

## 3. Implemented Kernels and Optimizations

### 3.1 CUDA Kernels

#### 3.1.1 Convolution Kernels

- **2D Convolution (Conv2D):**
    - **Naïve:** A straightforward implementation with high global memory traffic and redundant data loads.
    - **Tiled Shared Memory:** Divides the input and filter into tiles that are loaded into shared memory, reducing global memory accesses.
    - **Memory Coalescing:** Optimizes memory access patterns to ensure that threads in a warp access contiguous memory locations.
    - **Tensor Cores (WMMA):** Utilizes the `nvcuda::wmma` API to leverage Tensor Cores for the matrix multiplication at the heart of the convolution operation.

- **3D Convolution (Conv3D):**
    - **Naïve:** Similar to the 2D naïve implementation, but extended to three dimensions.
    - **Shared Memory:** Caches 3D data blocks in shared memory to improve data reuse.
    - **Tiled & Register Blocking:** Further reduces memory latency by blocking data in registers and tiles.

#### 3.1.2 Matrix & Reduction Operations

- **Matrix Transpose:**
    - **Naïve:** A simple implementation that suffers from non-coalesced memory accesses.
    - **Shared Memory Tiling:** Uses shared memory to perform the transpose in-place, enabling coalesced writes to global memory.

- **Matrix Multiplication (GEMM):**
    - **Naïve:** A basic implementation with O(N^3) complexity and high global memory traffic.
    - **Tiled:** Caches matrix tiles in shared memory to reduce global memory accesses.
    - **Register Blocking:** Uses registers to store a sub-matrix, further improving data reuse.
    - **Warp-Level Tiling:** Optimizes data exchange between threads at the warp level.
    - **Tensor Cores (WMMA):** Leverages Tensor Cores for maximum performance.

- **Reduction (Sum and Max):**
    - **Naïve:** A simple parallel reduction that suffers from thread divergence.
    - **Branchless Reduction:** Avoids thread divergence by using predicated execution.
    - **Warp-Level Reduction:** Uses warp shuffle intrinsics for a highly efficient, branchless reduction within a warp.
    - **Vectorized Reduction:** Uses vector types like `float4` to perform multiple reductions in parallel.

#### 3.1.3 Element-wise & Activation Functions

- **ReLU, Sigmoid, SwiGLU:**
    - **Naïve:** Simple element-wise operations.
    - **Coalesced Memory Access:** Ensures that memory accesses are coalesced.
    - **Vectorized Execution:** Uses vector types (`float4`, `float2`) to process multiple elements per thread.

- **SoftMax:**
    - **Naïve:** A basic implementation that is inefficient due to redundant memory accesses.
    - **Shared Memory Optimization:** Caches data in shared memory to reduce global memory traffic.
    - **Warp-Level Reduction:** Uses warp shuffle intrinsics for the reduction step.

#### 3.1.4 Image Processing Kernels

- **Greyscale Conversion & Image Blurring:**
    - These kernels demonstrate how to apply the same optimization principles (shared memory, coalescing, vectorization) to image processing tasks.

#### 3.1.5 Sorting Kernels

- **Bitonic Sort & Radix Sort:**
    - These kernels showcase how to implement classic sorting algorithms on the GPU.

### 3.2 Triton Kernels

Triton is a Python-based language and compiler for writing highly efficient GPU kernels. The Triton kernels in KernelLab provide a higher-level abstraction compared to CUDA, while still achieving competitive performance.

- **Vector Addition, Matrix Multiplication (GEMM), Softmax, Layer Normalization, GeGLU, RoPE Embedding, Flash Attention, SwiGLU:**
    - These kernels are implemented using Triton's high-level abstractions, which automatically handle many of the low-level optimization details.

## 4. Benchmarking and Performance

KernelLab includes a suite of benchmarks for comparing the performance of the different kernel implementations. The benchmarks are written in Python and use the `torch` library for creating and managing GPU tensors.

The results of the benchmarks are presented in the `README.md` file and in the `benchmarks` directory.

## 5. How to Use KernelLab

The CUDA kernels are exposed to Python via Pybind11. Each kernel has a `setup.py` file that can be used to build and install the Python bindings.

The Triton kernels can be used directly from Python.

## 6. Future Work

The `TODO.md` file lists the kernels and features that are planned for future development.