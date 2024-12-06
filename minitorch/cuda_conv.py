import numpy as np
from numba import cuda, float32
import math


# ------------------- CUDA Implementations -------------------

# Define CUDA kernels for conv1d and conv2d

@cuda.jit
def conv1d_kernel(input, weight, output, reverse, batch, in_channels, width, out_channels, k_width):
    """
    CUDA Kernel for 1D Convolution.

    Args:
        input (float32[:,:,:]): Input array on device.
        weight (float32[:,:,:]): Weight array on device.
        output (float32[:,:,:]): Output array on device.
        reverse (int): 1 if reverse else 0.
        batch (int): Number of batches.
        in_channels (int): Number of input channels.
        width (int): Width of the input.
        out_channels (int): Number of output channels.
        k_width (int): Kernel width.
    """
    b, oc, x = cuda.grid(3)  # 3D grid

    if b < batch and oc < out_channels and x < width:
        val = 0.0
        for ic in range(in_channels):
            for k in range(k_width):
                if reverse:
                    in_x = x - k
                else:
                    in_x = x + k
                if 0 <= in_x < width:
                    val += input[b, ic, in_x] * weight[oc, ic, k]
        output[b, oc, x] = val

@cuda.jit
def conv2d_kernel(input, weight, output, reverse, batch, in_channels, height, width, out_channels, k_height, k_width):
    """
    CUDA Kernel for 2D Convolution.

    Args:
        input (float32[:,:,:,:]): Input array on device.
        weight (float32[:,:,:,:]): Weight array on device.
        output (float32[:,:,:,:]): Output array on device.
        reverse (int): 1 if reverse else 0.
        batch (int): Number of batches.
        in_channels (int): Number of input channels.
        height (int): Height of the input.
        width (int): Width of the input.
        out_channels (int): Number of output channels.
        k_height (int): Kernel height.
        k_width (int): Kernel width.
    """
    b, oc, idx = cuda.grid(3)  # Now we only use 3D indexing
    if b < batch and oc < out_channels and idx < (height * width):
        # Decode h, w from idx
        h = idx // width
        w = idx % width

        val = 0.0
        for ic in range(in_channels):
            for kh in range(k_height):
                in_h = h - kh if reverse else h + kh
                if in_h < 0 or in_h >= height:
                    continue
                for kw in range(k_width):
                    in_w = w - kw if reverse else w + kw
                    if in_w < 0 or in_w >= width:
                        continue
                    val += input[b, ic, in_h, in_w] * weight[oc, ic, kh, kw]
        output[b, oc, h, w] = val

def cuda_conv1d(input_np, weight_np, reverse=False):
    """
    1D Convolution on GPU using CUDA.

    Args:
        input_np (np.ndarray): Input array of shape (batch, in_channels, width).
        weight_np (np.ndarray): Weight array of shape (out_channels, in_channels, k_width).
        reverse (bool): If True, reverse the kernel.

    Returns:
        np.ndarray: Output array of shape (batch, out_channels, width).
    """
    batch, in_channels, width = input_np.shape
    out_channels, in_channels2, k_width = weight_np.shape
    assert in_channels == in_channels2, "Input and weight in_channels must match."

    output_np = np.zeros((batch, out_channels, width), dtype=np.float32)

    # Transfer data to GPU
    input_gpu = cuda.to_device(input_np)
    weight_gpu = cuda.to_device(weight_np)
    output_gpu = cuda.to_device(output_np)

    # Define thread and block dimensions
    threads_per_block = (8, 8, 8)
    blocks_per_grid_x = math.ceil(batch / threads_per_block[0])
    blocks_per_grid_y = math.ceil(out_channels / threads_per_block[1])
    blocks_per_grid_z = math.ceil(width / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # Launch kernel
    conv1d_kernel[blocks_per_grid, threads_per_block](input_gpu, weight_gpu, output_gpu, int(reverse), batch, in_channels, width, out_channels, k_width)

    # Copy result back to CPU
    output_gpu.copy_to_host(output_np)

    return output_np

def cuda_conv2d(input_np, weight_np, reverse=False):
    """
    2D Convolution on GPU using CUDA.

    Args:
        input_np (np.ndarray): Input array of shape (batch, in_channels, height, width).
        weight_np (np.ndarray): Weight array of shape (out_channels, in_channels, k_height, k_width).
        reverse (bool): If True, reverse the kernel.

    Returns:
        np.ndarray: Output array of shape (batch, out_channels, height, width).
    """
    batch, in_channels, height, width = input_np.shape
    out_channels, in_channels2, k_height, k_width = weight_np.shape
    assert in_channels == in_channels2

    output_np = np.zeros((batch, out_channels, height, width), dtype=np.float32)

    input_gpu = cuda.to_device(input_np)
    weight_gpu = cuda.to_device(weight_np)
    output_gpu = cuda.to_device(output_np)

    # Flatten h and w
    total_pixels = height * width

    threads_per_block = (8, 8, 8)  # for example
    blocks_per_grid_b = math.ceil(batch / threads_per_block[0])
    blocks_per_grid_oc = math.ceil(out_channels / threads_per_block[1])
    blocks_per_grid_hw = math.ceil(total_pixels / threads_per_block[2])
    blocks_per_grid = (blocks_per_grid_b, blocks_per_grid_oc, blocks_per_grid_hw)

    conv2d_kernel[blocks_per_grid, threads_per_block](
        input_gpu,
        weight_gpu,
        output_gpu,
        int(reverse),
        batch,
        in_channels,
        height,
        width,
        out_channels,
        k_height,
        k_width
    )

    output_gpu.copy_to_host(output_np)
    return output_np

# ------------------- Testing Functions -------------------

def test_conv1d():
    print("Testing 1D Convolution...")

    # Test Case 1 (Non-reverse)
    print("\nTest Case 1:")
    input_np = np.random.randn(2, 3, 5).astype(np.float32)
    weight_np = np.random.randn(4, 3, 3).astype(np.float32)
    cpu_out = cpu_conv1d(input_np, weight_np, reverse=False)
    gpu_out = cuda_conv1d(input_np, weight_np, reverse=False)
    difference = np.max(np.abs(cpu_out - gpu_out))
    print(f"Conv1D Test 1 - Max difference: {difference}")

    # Test Case 2 (Reverse)
    print("\nTest Case 2 (Reverse):")
    input_np = np.random.randn(1, 2, 10).astype(np.float32)
    weight_np = np.random.randn(2, 2, 5).astype(np.float32)
    cpu_out_reverse = cpu_conv1d(input_np, weight_np, reverse=True)
    gpu_out = cuda_conv1d(input_np, weight_np, reverse=True)
    difference = np.max(np.abs(cpu_out_reverse - gpu_out))
    print(f"Conv1D Test 2 (Reverse) - Max difference: {difference}")

def test_conv2d():
    print("\nTesting 2D Convolution...")

    # Test Case 1 (Non-reverse)
    print("\nTest Case 1:")
    input_np = np.random.randn(2, 3, 7, 7).astype(np.float32)
    weight_np = np.random.randn(4, 3, 3, 3).astype(np.float32)
    cpu_out = cpu_conv2d(input_np, weight_np, reverse=False)
    gpu_out = cuda_conv2d(input_np, weight_np, reverse=False)
    difference = np.max(np.abs(cpu_out - gpu_out))
    print(f"Conv2D Test 1 - Max difference: {difference}")

    # Test Case 2 (Reverse)
    print("\nTest Case 2 (Reverse):")
    input_np = np.random.randn(1, 2, 5, 5).astype(np.float32)
    weight_np = np.random.randn(2, 2, 2, 2).astype(np.float32)
    cpu_out_reverse = cpu_conv2d(input_np, weight_np, reverse=True)
    gpu_out = cuda_conv2d(input_np, weight_np, reverse=True)
    difference = np.max(np.abs(cpu_out_reverse - gpu_out))
    print(f"Conv2D Test 2 (Reverse) - Max difference: {difference}")

# ------------------- Run Tests -------------------

# Execute the tests
test_conv1d()
test_conv2d()

# Output:

# Test Case 1:
# Conv1D Test 1 - Max difference: 0.0

# Test Case 2 (Reverse):
# Conv1D Test 2 (Reverse) - Max difference: 0.0

# Testing 2D Convolution...

# Test Case 1:
# Conv2D Test 1 - Max difference: 0.0

# Test Case 2 (Reverse):
# Conv2D Test 2 (Reverse) - Max difference: 0.0