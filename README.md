# MiniTorch Module 4

This codebase is part of the MiniTorch project, a DIY deep learning framework designed for educational purposes. Module 4 focuses on implementing convolutional neural networks (CNNs) with an emphasis on efficiency and utilizing Numba for just-in-time (JIT) compilation to achieve performance gains. Key components include:

*   **Tensor Operations and Autodiff**: Core implementations for tensor manipulation, including broadcasting, mapping, zipping, and reductions. Automatic differentiation is supported for gradient-based optimization.
*   **CUDA Support**: (Optional) Enables GPU acceleration using CUDA, particularly for computationally intensive operations like matrix multiplication and convolutions.
*   **Neural Network Building Blocks**: Includes implementations for common neural network layers, such as linear (fully connected) layers, and 1D/2D convolutional layers.
*   **Optimizers**: Contains optimization algorithms, such as Stochastic Gradient Descent (SGD), to update model parameters during training.
*   **CNN Implementation**: Features a CNN model based on the architecture described in Y. Kim 2014, suitable for sentiment analysis tasks.
*   **Datasets**: Utilities for generating synthetic datasets for testing, and loaders for the MNIST and GLUE SST2 datasets.
*   **Training Scripts**: Example training scripts for image classification (MNIST) and sentiment analysis (GLUE SST2) tasks.
*   **Visualization Tools**: Provides functionalities for visualizing tensor data structures, computational graphs, and training progress using Plotly and Graphviz.
*   **Interactive Interfaces**: Includes Streamlit-based interfaces for exploring tensor operations, visualizing module trees, and running training experiments with interactive parameter tuning.

## Key Files

*   `minitorch/`: Contains the core library code.
    *   `tensor.py`: Implements the `Tensor` class with support for automatic differentiation.
    *   `tensor_ops.py`: Defines basic tensor operations.
    *   `tensor_functions.py`: Includes higher-level tensor functions and operations.
    *   `tensor_data.py`: Manages underlying tensor data storage, strides, and indexing.
    *   `fast_ops.py`: Provides Numba-accelerated implementations of tensor operations.
    *   `cuda_ops.py`: Offers CUDA-accelerated tensor operations.
    *   `cuda_conv.py`: Contains CUDA implementations for 1D and 2D convolutions.
    *   `nn.py`: Implements neural network layers and pooling operations.
    *   `optim.py`: Includes optimizers for training neural networks.
    *   `module.py`: Defines the `Module` class for building neural network models.
    *   `autodiff.py`: Handles automatic differentiation logic.
    *   `datasets.py`: Utilities for generating and loading datasets.
    *   `operators.py`: Contains implementations of basic mathematical operators.
    *   `scalar.py`, `scalar_functions.py`: Implement scalar operations and their derivatives for autodiff.
    *   `testing.py`: Utilities for testing.

*   `project/`: Contains scripts and modules for running experiments and interactive interfaces.
    *   `run_manual.py`: Example script for training a simple network manually.
    *   `run_mnist_multiclass.py`: Script for training a CNN on the MNIST dataset.
    *   `run_mnist_interface.py`: Streamlit interface for MNIST training and visualization.
    *   `run_sentiment.py`: Script for training a CNN on the GLUE SST2 sentiment analysis task.
    *   `sentiment_interface.py`: Streamlit interface for sentiment analysis training and visualization.
    *   `run_tensor.py`: Script for training a model with tensor operations.
    *   `run_torch.py`: Example script for training a neural network using PyTorch.
    *   `math_interface.py`: Streamlit interface for exploring mathematical functions and their derivatives.
    *   `module_interface.py`: Streamlit interface for visualizing module trees.
    *   `tensor_interface.py`: Streamlit interface for exploring tensor operations and visualizations.
    *   `graph_builder.py`: Utility for building computational graphs.
    *   `app.py`: Main Streamlit application file that ties together all the interfaces.
    *   `data/`: Directory containing the MNIST dataset files.
    *   `embeddings.py`: This is part of a third-party library and handles word embeddings.
    *   `requirements.extra.txt`: Lists additional Python dependencies.
## Setup

To set up the environment for this project, you can use the provided `environment.yml` file with Conda:

```bash
conda env create -f environment.yml
conda activate minitorch

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py
