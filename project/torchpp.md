# TorchPP: The PyTorch Performance Plus Toolkit

`torchpp` is a powerful extension library for PyTorch, designed to supercharge your deep learning workflows. It provides a suite of high-performance CUDA kernels and a powerful distributed training framework to accelerate model performance and simplify scaling.

Whether you are working with **Large Language Models (LLMs)**, **Diffusion Models**, **Text-to-Speech (TTS)**, or **Time-Series Models**, `torchpp` aims to be your go-to library for performance and scalability.

## Core Pillars

### 1. Accelerate Your Models

Boost your model's speed with our collection of high-performance, custom-written CUDA kernels. We leverage libraries like **CUTLASS** to build fused and optimized components that replace standard PyTorch modules, resulting in significant performance gains by reducing memory bandwidth and leveraging hardware-specific features like Tensor Cores.

*   **Fused Kernels:**
    *   **Linear + Activation:** Fused `Linear` layers with `GeLU` and `SiLU` activations.
    *   **Optimized Normalization:** High-performance `LayerNorm` and `RMSNorm`.
*   **Custom Implementations:**
    *   **Rotary Position Embeddings (RoPE):** An optimized implementation of RoPE.
*   **Attention Variants:** Efficient implementations of various attention mechanisms, including `Multi-Head`, `Multi-Query`, `Grouped-Query`, `Sliding-Window`, and `Cross-Attention`, all using `flash-attn` for maximum performance.

### 2. Simplify Distributed Training

Move beyond the boilerplate of distributed training. `torchpp` provides a high-level, easy-to-use `DistributedTrainer` that handles the complexities of different parallelization strategies, so you can focus on your model.

*   **Effortless Scaling:** Easily switch between strategies like Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and hybrid approaches with simple configuration changes.
*   **Out-of-the-Box Functionality:** The trainer includes built-in support for mixed-precision training, gradient accumulation, checkpointing, and more.

## How it Works: The `torchpp` Architecture

`torchpp` achieves its performance by combining a low-level CUDA backend with a high-level Python frontend.

1.  **CUDA Kernels (`csrc/`):** The core operations are written in C++ and CUDA. These kernels are optimized for specific hardware (e.g., NVIDIA GPUs with Tensor Cores) and data types (currently `FP16`).
2.  **Pybind11 Bindings:** We use `pybind11` to create Python bindings for the C++ functions. This is handled in the `binding.cu` files within the `csrc` directory (e.g., `csrc/activation/binding.cu`). These bindings compile the CUDA code into Python modules (e.g., `linearActvationFp16`).
3.  **Python API (`torchpp/`):** The user-facing API is written in Python. The modules in `torchpp/` (e.g., `torchpp.dlops.linear`) import the compiled CUDA modules and wrap them in familiar `torch.nn.Module` classes, making them easy to integrate into existing PyTorch models.

This architecture allows you to get the performance of low-level CUDA programming with the ease of use of a high-level Python library.

## Installation

**Prerequisites:**

*   A CUDA-enabled GPU.
*   The [CUTLASS library](https://github.com/NVIDIA/cutlass). Ensure the `CUTLASS_PATH` environment variable is set.

```bash
export CUTLASS_PATH=/path/to/cutlass/include
```

**Installation:**

```bash
git clone https://github.com/AmanSwar/TorchPlusPlus.git
cd torchpp
pip install .
```

---

## API Reference

### Deep Learning Operations (`torchpp.dlops`)

This module provides optimized implementations of common deep learning operations. **All modules in `torchpp.dlops` currently expect `FP16` tensors.**

#### Normalization (`torchpp.dlops.normalization`)

**`RmsNormFused(nn.Module)`**

A high-performance implementation of Root Mean Square Normalization.

*   **Arguments:**
    *   `normalize_dim_shape` (int): The size of the dimension to normalize.
    *   `eps` (float, optional): A small value to avoid division by zero. Defaults to `1e-6`.
    *   `dtype` (torch.dtype, optional): The data type of the weight. Defaults to `torch.float16`.
    *   `device` (torch.device, optional): The device of the weight. Defaults to `torch.device("cuda")`.
*   **Usage:**
    ```python
    import torch
    from torchpp.dlops.normalization import RmsNormFused

    norm = RmsNormFused(normalize_dim_shape=256)
    x = torch.randn(16, 128, 256, dtype=torch.float16, device="cuda")
    output = norm(x)
    ```

**`LayerNormFused(nn.Module)`**

A high-performance implementation of Layer Normalization.

*   **Arguments:**
    *   `normalize_dim_shape` (int): The size of the dimension to normalize.
    *   `eps` (float, optional): A small value to avoid division by zero. Defaults to `1e-6`.
    *   `dtype` (torch.dtype, optional): The data type of the weight. Defaults to `torch.float16`.
    *   `device` (torch.device, optional): The device of the weight. Defaults to `torch.device("cuda")`.
*   **Usage:**
    ```python
    import torch
    from torchpp.dlops.normalization import LayerNormFused

    norm = LayerNormFused(normalize_dim_shape=256)
    x = torch.randn(16, 128, 256, dtype=torch.float16, device="cuda")
    output = norm(x)
    ```

#### Fused Linear Layers (`torchpp.dlops.linear`)

**`LinearGELU(nn.Module)`**

A `Linear` layer fused with a `GELU` activation.

*   **Arguments:**
    *   `in_features` (int): Size of each input sample.
    *   `out_features` (int): Size of each output sample.
*   **Usage:**
    ```python
    import torch
    from torchpp.dlops.linear import LinearGELU

    layer = LinearGELU(in_features=512, out_features=1024)
    x = torch.randn(32, 512, dtype=torch.float16, device="cuda")
    output = layer(x)
    ```

**`LinearSILU(nn.Module)`**

A `Linear` layer fused with a `SiLU` (Swish) activation.

*   **Arguments:**
    *   `in_features` (int): Size of each input sample.
    *   `out_features` (int): Size of each output sample.
*   **Usage:**
    ```python
    import torch
    from torchpp.dlops.linear import LinearSILU

    layer = LinearSILU(in_features=512, out_features=1024)
    x = torch.randn(32, 512, dtype=torch.float16, device="cuda")
    output = layer(x)
    ```

#### Rotary Position Embeddings (`torchpp.dlops.rope`)

**`rope_apply(x, cos, sin)`**

A functional implementation of Rotary Position Embeddings.

*   **Arguments:**
    *   `x` (torch.Tensor): The input tensor of shape `[batch, heads, seq_len, head_dim]`.
    *   `cos` (torch.Tensor): The cosine cache of shape `[seq_len, head_dim]`.
    *   `sin` (torch.Tensor): The sine cache of shape `[seq_len, head_dim]`.
*   **Returns:** A `torch.Tensor` with RoPE applied.
*   **Usage:**
    ```python
    import torch
    from torchpp.dlops.rope import rope_apply

    # Assume x, cos_cache, and sin_cache are precomputed
    # x: [bs, n_heads, seq_len, head_dim]
    # cos_cache, sin_cache: [seq_len, head_dim]
    output = rope_apply(x, cos_cache, sin_cache)
    ```

### Attention Mechanisms (`torchpp.attention`)

This module provides several efficient attention implementations that leverage `flash-attn`.

*   **`MultiHeadAttention(embed_dim, n_heads, ...)`**: Standard Multi-Head Attention.
*   **`GroupedQueryAttention(d_in, num_heads, n_kv_heads, ...)`**: Grouped-Query Attention.
*   **`MQA_FA(num_q_heads, embed_dim, ...)`**: Multi-Query Attention.
*   **`SlidingWindowAttention(window_size, embed_dim, n_heads, ...)`**: Sliding Window Attention.
*   **`CrossAttention(embed_dim, cross_dim, n_heads, ...)`**: Cross-Attention.

All attention modules follow a similar pattern and are initialized with model dimensions and optional arguments like `qknorm` and `dtype`.

### Distributed Training (`torchpp.train.dist_train`)

**`DistributedTrainer`**

A high-level trainer class to handle the complexities of distributed training.

*   **Key Arguments:**
    *   `model` (nn.Module): The model to be trained.
    *   `config` (TrainingConfig): A dataclass containing all training configurations (strategy, learning rate, checkpoint paths, etc.).
    *   `optimizer` (torch.optim.Optimizer, optional): The optimizer to use. If not provided, an `AdamW` optimizer is created by default.
    *   `lr_sched` (optional): A learning rate scheduler.
    *   `loss_function` (Callable, optional): The loss function. Defaults to cross-entropy loss.
*   **Key Methods:**
    *   `train(train_dataloader, eval_dataloader, num_epochs)`: Starts the training loop.
    *   `evaluate(eval_dataloader)`: Runs the evaluation loop.
    *   `_save_checkpoint()`: Saves a model checkpoint.
    *   `load_checkpoint(path)`: Loads a model checkpoint.

## Future Vision & Roadmap

We have an ambitious roadmap to make `torchpp` an indispensable tool for PyTorch developers:

1.  **Quantization Support:** Integration of popular quantization techniques like AWQ, GPTQ, and others to further boost inference performance.
2.  **Faster Training with Custom Backward Kernels:** Implementation of custom backward passes for all our fused kernels to accelerate the training process.
3.  **Expanded Kernel Library:** Introduction of new fused kernels for Diffusion, Convolution-based models, and RNN-based models.
