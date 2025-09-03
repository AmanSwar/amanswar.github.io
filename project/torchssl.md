
### 1. Introduction

TorchSSL is a PyTorch-based library designed to provide a clean, modular, and high-performance environment for self-supervised learning with visual representations. The library is structured to be easy to use for both research and development, with a focus on extensibility and performance.

### 2. Core Components

The library is organized into several core components:

*   `torchssl/framework`: Contains the main SSL algorithm implementations (SimCLR, MoCo, DINO, I-JEPA).
*   `torchssl/model`: Includes backbone models and projectors.
*   `torchssl/loss`: Provides loss functions, including custom Triton-based implementations.
*   `torchssl/dataset`: Contains data loading and augmentation pipelines.
*   `torchssl/eval`: Includes evaluation methods like kNN and Linear Probing.

### 3. Frameworks

Each SSL framework is implemented as a class that inherits from the base `SSL` class. The main frameworks are:

*   **`SimCLR`**: Implements the SimCLR framework. It uses a `SimclrModel` which consists of a backbone and a projection head. The training process involves generating two augmented views of each image and minimizing the NT-Xent loss between their representations.

*   **`MoCO`**: Implements the Momentum Contrast (MoCo) framework. It uses a `MocoModel` with a query encoder, a key encoder, and a queue of negative samples. The key encoder is a momentum-updated version of the query encoder. The loss is calculated using the InfoNCE loss.

*   **`Dino`**: Implements the DINO framework. It uses a `DinoModel` with a student and a teacher network. The teacher's weights are an exponential moving average of the student's weights. The loss is calculated between the student's and teacher's outputs, encouraging the student to learn from the teacher.

*   **`IJEPA`**: Implements the Image-based Joint-Embedding Predictive Architecture (I-JEPA). It uses a context encoder and a target encoder. The goal is to predict the representations of target blocks in an image from a given context block.

### 4. Loss Functions

TorchSSL includes both standard PyTorch and high-performance Triton-based loss functions:

*   **`NTXentLoss` and `NTXentLossTriton`**: The Normalized Temperature-scaled Cross-Entropy loss used in SimCLR. The Triton version is optimized for performance.
*   **`InfoNCELoss` and `InfoNCELossTriton`**: The InfoNCE loss used in MoCo. The Triton version provides a significant speedup.
*   **`DINOLoss`**: The loss function for the DINO framework, which involves a cross-entropy loss between the student and teacher outputs.
*   **`IJEPALoss`**: The loss for I-JEPA, which is a mean squared error between the predicted and target features.

### 5. Data Loading and Augmentation

*   **`SSLDataloader`**: A flexible data loader that can handle training and validation splits from a single directory or separate directories. It wraps the `SSLDataset` class.
*   **`sslaug.py`**: Contains data augmentation pipelines for each SSL framework:
    *   `SimclrAug`: Augmentations for SimCLR.
    *   `MocoAug`: Augmentations for MoCo.
    *   `DinoAug`: Multi-crop augmentations for DINO.
    *   `IjepaAug`: Augmentations for I-JEPA.

### 6. Models and Backbones

*   **`Backbone`**: A wrapper class for `timm` models, allowing for easy integration of various architectures like `ConvNeXt` and `ResNet`.
*   The framework-specific model classes (`SimclrModel`, `MocoModel`, `DinoModel`, `IjepaModel`) combine the backbone with the necessary projection heads or other components.

### 7. Evaluation

The `EvaluateSSL` class in `torchssl/eval/Eval.py` provides methods for evaluating the quality of the learned representations:

*   **`linear_probe_evaluation`**: Trains a linear classifier on top of the frozen features of the pre-trained model.
*   **`knn_evaluation`**: Performs k-Nearest Neighbors classification on the learned features.
*   **`monitor_feature_representation`**: Provides tools for visualizing the latent space using PCA and t-SNE, and for logging feature statistics to WandB.

### 8. Examples

The `examples/` directory provides scripts to demonstrate how to use the different SSL frameworks. These scripts show how to set up the data loader, model, optimizer, and training loop for each framework.
