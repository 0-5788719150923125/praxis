import torch
import triton
import triton.language as tl


@triton.jit
def rnn_forward_kernel(
    # Pointers to matrices
    x_ptr,
    h_ptr,
    w_ih_ptr,
    w_hh_ptr,
    bias_ih_ptr,
    bias_hh_ptr,  # Now two bias terms
    output_ptr,
    # Matrix dimensions
    batch_size,
    seq_len,
    input_size,
    hidden_size,
    # Strides for the different matrices
    stride_x_batch,
    stride_x_seq,
    stride_x_input,
    stride_h_batch,
    stride_h_hidden,
    stride_out_batch,
    stride_out_seq,
    stride_out_hidden,
    # Block sizes for tiling
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)

    # Compute batch and sequence indices
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Bounds checking
    if batch_idx >= batch_size:
        return

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Get input block
    x_block_ptr = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    x = tl.load(
        x_block_ptr
        + tl.arange(0, BLOCK_M)[:, None] * stride_x_input
        + tl.arange(0, BLOCK_K)[None, :]
    )

    # Load weights with correct strides for [hidden_size, input_size] layout
    w_ih_block = tl.load(
        w_ih_ptr
        + tl.arange(0, BLOCK_N)[:, None]
        + tl.arange(0, BLOCK_K)[None, :] * hidden_size
    )

    # First term: x @ w_ih.T + bias_ih
    acc += tl.dot(x, w_ih_block.T)

    # Load and add input-hidden bias
    bias_ih = tl.load(bias_ih_ptr + tl.arange(0, BLOCK_N))
    acc += bias_ih[None, :]

    # Handle previous hidden state
    if seq_idx > 0:
        h_prev_offset = batch_idx * stride_h_batch + (seq_idx - 1) * stride_h_hidden
        h_prev = tl.load(
            h_ptr
            + h_prev_offset
            + tl.arange(0, BLOCK_M)[:, None] * stride_h_hidden
            + tl.arange(0, BLOCK_K)[None, :]
        )

        # Load hidden weights
        w_hh_block = tl.load(
            w_hh_ptr
            + tl.arange(0, BLOCK_N)[:, None]
            + tl.arange(0, BLOCK_K)[None, :] * hidden_size
        )

        # Second term: h_prev @ w_hh.T + bias_hh
        acc += tl.dot(h_prev, w_hh_block.T)

    # Load and add hidden-hidden bias
    bias_hh = tl.load(bias_hh_ptr + tl.arange(0, BLOCK_N))
    acc += bias_hh[None, :]

    # Apply tanh activation
    pos_exp = tl.exp(acc)
    neg_exp = tl.exp(-acc)
    acc = (pos_exp - neg_exp) / (pos_exp + neg_exp)

    # Write output
    out_offset = batch_idx * stride_out_batch + seq_idx * stride_out_seq
    tl.store(
        output_ptr
        + out_offset
        + tl.arange(0, BLOCK_M)[:, None] * stride_out_hidden
        + tl.arange(0, BLOCK_N)[None, :],
        acc,
    )


class TritonRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # Initialize weights to match PyTorch's layout
        self.w_ih = torch.nn.Parameter(
            torch.randn(hidden_size, input_size) / (input_size**0.5)
        )
        self.w_hh = torch.nn.Parameter(
            torch.randn(hidden_size, hidden_size) / (hidden_size**0.5)
        )
        # Two separate bias terms to match PyTorch
        self.bias_ih = torch.nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_0=None):
        if self.batch_first:
            batch_size, seq_len, _ = x.shape
        else:
            seq_len, batch_size, _ = x.shape
            x = x.transpose(0, 1)

        device = x.device

        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)

        # Prepare output tensor
        output = torch.empty(batch_size, seq_len, self.hidden_size, device=device)

        # Calculate block sizes
        BLOCK_M = 32  # batch dimension
        BLOCK_N = 32  # hidden dimension
        BLOCK_K = 32  # input dimension

        # Launch kernel
        grid = lambda meta: (batch_size * seq_len,)
        rnn_forward_kernel[grid](
            x.contiguous(),
            h_0.contiguous(),
            self.w_ih.contiguous(),
            self.w_hh.contiguous(),
            self.bias_ih.contiguous(),  # Pass both bias terms
            self.bias_hh.contiguous(),
            output,
            batch_size,
            seq_len,
            self.input_size,
            self.hidden_size,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            h_0.stride(0),
            h_0.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, h_0


if __name__ == "__main__":
    # Test parameters
    batch_size = 4  # Smaller batch for debugging
    seq_length = 2  # Shorter sequence for debugging
    input_size = 8  # Smaller input for debugging
    hidden_size = 16  # Smaller hidden for debugging

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create random input
    x = torch.randn(batch_size, seq_length, input_size).cuda()

    # Initialize model
    model = TritonRNN(input_size, hidden_size).cuda()

    # PyTorch implementation
    torch_rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True).cuda()

    # Copy weights for fair comparison
    with torch.no_grad():
        print("\nWeight shapes:")
        print(f"Triton w_ih: {model.w_ih.shape}")
        print(f"PyTorch weight_ih_l0: {torch_rnn.weight_ih_l0.shape}")
        print(f"Triton w_hh: {model.w_hh.shape}")
        print(f"PyTorch weight_hh_l0: {torch_rnn.weight_hh_l0.shape}")

        # Copy and verify weights
        torch_rnn.weight_ih_l0.copy_(model.w_ih)
        torch_rnn.weight_hh_l0.copy_(model.w_hh)
        torch_rnn.bias_ih_l0.copy_(model.bias_ih)
        torch_rnn.bias_hh_l0.copy_(model.bias_hh)

        print("\nWeight equality checks:")
        print(f"w_ih match: {torch.allclose(model.w_ih, torch_rnn.weight_ih_l0)}")
        print(f"w_hh match: {torch.allclose(model.w_hh, torch_rnn.weight_hh_l0)}")
        print(f"bias_ih match: {torch.allclose(model.bias_ih, torch_rnn.bias_ih_l0)}")
        print(f"bias_hh match: {torch.allclose(model.bias_hh, torch_rnn.bias_hh_l0)}")

    # Forward pass
    output, final_hidden = model(x)
    torch_output, torch_hidden = torch_rnn(x)

    # Print shape information
    print("\nOutput shapes:")
    print(f"Triton output: {output.shape}")
    print(f"PyTorch output: {torch_output.shape}")

    # Compare first timestep outputs
    print("\nFirst timestep comparison:")
    print("Triton first timestep:")
    print(output[0, 0, :8].cpu().detach().numpy())
    print("PyTorch first timestep:")
    print(torch_output[0, 0, :8].cpu().detach().numpy())

    # Calculate and print max difference
    max_diff = (output - torch_output).abs().max().item()
    print(f"\nMaximum absolute difference: {max_diff}")

    # Check closeness with different tolerances
    for rtol, atol in [(1e-3, 1e-3), (1e-2, 1e-2), (1e-1, 1e-1)]:
        is_close = torch.allclose(output, torch_output, rtol=rtol, atol=atol)
        print(f"Close with rtol={rtol}, atol={atol}: {is_close}")

    print("\nTest complete!")
