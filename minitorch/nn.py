from typing import Tuple

from .autodiff import Context
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    # Ensure the input is contiguous
    input = input.contiguous()

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    # Reshape: (B, C, H, W) -> (B, C, new_height, kh, new_width, kw)
    out = input.view(batch, channel, new_height, kh, new_width, kw)

    # Now we want (B, C, new_height, new_width, kh*kw)
    out = (
        out.permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return out, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform average 2D pooling on the input tensor using the provided kernel.
    The input is of shape (batch, channel, height, width).

    Args:
    ----
        input: The input tensor of shape (batch, channel, height, width).
        kernel: The (kernel_height, kernel_width) of the pooling operation.

    Returns:
    -------
        A Tensor of shape (batch, channel, new_height, new_width)

    """
    kh, kw = kernel
    tiled, new_height, new_width = tile(input, kernel)
    # After tiling: (batch, channel, new_height, new_width, kh*kw)

    # Summing over the last dimension reduces to shape: (batch, channel, new_height, new_width, 1)
    out = tiled.sum(dim=4) / (kh * kw)

    # Now reshape the output to remove the last dimension (size 1)
    batch, channel = input.shape[0], input.shape[1]
    out = out.view(batch, channel, new_height, new_width)

    return out


def unsqueeze(t: Tensor, dim: int) -> Tensor:
    """Add a dimension of size 1 at position dim."""
    shape = list(t.shape)  # Convert to list to allow concatenation
    new_shape = tuple(shape[:dim] + [1] + shape[dim:])  # Convert back to tuple
    return t.view(*new_shape)


def max_reduce_along_dim(input: Tensor, dim: int) -> Tensor:
    """Manually compute the max of `input` along `dim`."""
    shape = list(input.shape)  # Convert to list
    out_shape = tuple(shape[:dim] + [1] + shape[dim + 1 :])  # Convert back to tuple
    out = input.zeros(out_shape)

    for out_idx in out._tensor.indices():
        max_val = -float("inf")
        full_idx = list(out_idx[:dim]) + [0] + list(out_idx[dim + 1 :])
        for d in range(shape[dim]):
            full_idx[dim] = d
            val = input[tuple(full_idx)]
            if val > max_val:
                max_val = val
        out[out_idx] = max_val
    return out


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim_tensor: Tensor) -> Tensor:
        """Compute the maximum values along the dimension specified in dim_tensor."""
        ctx.save_for_backward(input, dim_tensor)
        dim = int(dim_tensor.item())
        out_vals = max_reduce_along_dim(input, dim)
        return out_vals

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute gradients for the max operation using saved values from forward pass."""
        (input, max_red) = ctx.saved_values
        return (grad_output * (max_red == input)), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along `dim`."""
    return Max.apply(input, tensor([float(dim)]))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a one-hot vector along a given dimension.
    Returns a tensor of the same shape as input with one dimension reduced,
    where the position of the maximum value is 1 and all others 0.

    """
    max_vals = max(input, dim)
    expanded_max = unsqueeze(max_vals, dim)
    return (input == expanded_max) * 1.0


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor."""
    # TODO: Implement for Task 4.4.
    t = input.exp()
    s = t.sum(dim)
    return t / s


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor."""
    softmax_tensor = softmax(input, dim)
    return softmax_tensor.log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform max 2D pooling:
    Similar to avgpool2d, but take the max instead of average.
    (batch, channel, new_height, new_width, kh*kw) -> max over last dimension
    """
    tiled, new_height, new_width = tile(input, kernel)
    batch, channel = input.shape[0], input.shape[1]

    # Take max over last dimension and reshape
    out = max(tiled, 4)  # dim=4 is the last dimension with pooled values
    return out.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Randomly zeroes out elements of the input tensor with probability p.
    If ignore=True, then do nothing (no dropout).

    """
    if ignore or p == 0.0:
        return input
    shape = input.shape
    total = 1
    for s in shape:
        total *= s
    r = rand((total,))
    r = r.view(*shape)
    mask = (r > p) * 1.0
    return input * mask
