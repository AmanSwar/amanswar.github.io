
### 1. Introduction

DistJax is a powerful and flexible library for JAX that simplifies the implementation of distributed training for large-scale neural networks. It provides high-level abstractions and reusable components for common parallelism strategies, enabling you to scale your models with minimal code changes.

**Key Features:**

*   **Data Parallelism:** Train your model on multiple devices with data sharding.
*   **Tensor Parallelism:** Shard your model's weights across multiple devices to train models that don't fit on a single device.
*   **Pipeline Parallelism:** Partition your model's layers across multiple devices to train very deep models efficiently.
*   **Asynchronous Communication:** Overlap communication and computation to maximize hardware utilization.
*   **Composable Primitives:** Mix and match different parallelism strategies to create hybrid solutions.

### 2. Core Concepts

#### a. Device Mesh

A core concept in DistJax is the **device mesh**, which is a logical grid of devices (e.g., GPUs or TPUs) that you can use to distribute your model and data. The mesh is defined by a shape and a set of axis names, which you can use to specify how your data and model are sharded.

```python
from jax.experimental.maps import Mesh
import numpy as np

num_devices = 8
# Create a 1D mesh for data parallelism
dp_mesh = Mesh(np.arange(num_devices).reshape(num_devices,), ('data',))

# Create a 2D mesh for data and tensor parallelism
tp_mesh = Mesh(np.arange(num_devices).reshape(4, 2), ('data', 'tensor'))
```

#### b. Sharding

DistJax uses `shard_map` and `PartitionSpec` to control how your data and model are sharded across the device mesh. A `PartitionSpec` is a tuple that specifies how each dimension of a tensor is sharded across the mesh axes.

```python
from jax.experimental.maps import PartitionSpec as P

# Replicate the weights across the 'data' axis
replicated = P()

# Shard the weights along the 'tensor' axis
sharded = P('tensor',)
```

### 3. Parallelism Strategies

#### a. Data Parallelism

Data parallelism is the most common and straightforward way to distribute your training. In this strategy, you replicate your model on each device and feed each device a different shard of the input data. Gradients are then averaged across all devices to update the model's weights.

**Example:**

```python
from DistJax.parallelism.data_parallel import data_parallel_train_step
from DistJax.models.simple_classifier import SimpleClassifier

# Define your model
model = SimpleClassifier(num_classes=10)

# Create a data-parallel training step
train_step = data_parallel_train_step(model)

# Run the training step on sharded data
sharded_train_step = shard_map(
    train_step,
    mesh=dp_mesh,
    in_specs=(P('data',), P('data',)),
    out_specs=P()
)
```

#### b. Tensor Parallelism

Tensor parallelism is a model parallelism technique where you shard the model's weights across multiple devices. This allows you to train models that are too large to fit on a single device. DistJax provides both synchronous and asynchronous tensor parallelism.

**Synchronous Tensor Parallelism:**

In synchronous tensor parallelism, communication is performed using collective operations like `all_gather` and `psum_scatter`.

**Asynchronous Tensor Parallelism:**

Asynchronous tensor parallelism overlaps communication and computation to hide communication latency. This is achieved using JAX's `ppermute` operation to pass activations between devices in a ring-like fashion.

**Example:**

```python
from DistJax.parallelism.tensor_parallel import TPDense
from DistJax.parallelism.tensor_parallel_async import TPAsyncDense

# Synchronous tensor-parallel dense layer
dense_layer = TPDense(
    features=1024,
    kernel_init=jax.nn.initializers.glorot_normal(),
    mesh=tp_mesh,
)

# Asynchronous tensor-parallel dense layer
async_dense_layer = TPAsyncDense(
    features=1024,
    kernel_init=jax.nn.initializers.glorot_normal(),
    mesh=tp_mesh,
)
```

#### c. Pipeline Parallelism

Pipeline parallelism is another model parallelism technique where you partition the layers of your model across multiple devices. The input batch is split into micro-batches, which are fed into the pipeline in a staggered manner to keep all devices active.

**Example:**

```python
from DistJax.parallelism.pipeline_parallel import pipeline_parallel_train_step
from DistJax.models.pp_classifier import PPClassifier

# Define your model
model = PPClassifier(num_classes=10, num_layers=4)

# Create a pipeline-parallel training step
train_step = pipeline_parallel_train_step(model, num_micro_batches=4)

# Run the training step
sharded_train_step = shard_map(
    train_step,
    mesh=pp_mesh,
    in_specs=(P('data',), P('data',)),
    out_specs=P()
)
```

### 4. Models

DistJax provides several example models that demonstrate how to use the different parallelism strategies:

*   **`SimpleClassifier`:** A basic classifier for demonstrating data parallelism.
*   **`TPClassifier`:** A classifier that uses tensor parallelism.
*   **`PPClassifier`:** A classifier that uses pipeline parallelism.
*   **`Transformer`:** A Transformer model with tensor parallelism.

### 5. Getting Started

To get started with DistJax, please refer to the `README.md` file for installation instructions and examples.

### 6. API Reference

The core components of DistJax are located in the `DistJax/parallelism` directory:

*   **`data_parallel.py`:** Data parallelism primitives.
*   **`tensor_parallel.py`:** Synchronous tensor parallelism primitives.
*   **`tensor_parallel_async.py`:** Asynchronous tensor parallelism primitives.
*   **`pipeline_parallel.py`:** Pipeline parallelism primitives.
*   **`sharding.py`:** Utilities for sharding.

### 7. Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on our GitHub repository.