# https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes
# TODO: slower than fp16

import torch
import triton
from quantize_rowwise import quantize_rowwise
from dequantize_rowwise import dequantize_rowwise
from int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize

def matmul(X, W):
    X_int8, state_X = quantize_rowwise(X)
    W_int8, state_W = quantize_rowwise(W)

    return int8_matmul_rowwise_dequantize(X_int8, W_int8.t(), state_X, state_W, bias = None)

torch.manual_seed(0)
# M, N, K = 32, 256, 1024
# a = torch.randn((M, N), device='cuda', dtype=torch.float16)
# b = torch.randn((N, K), device='cuda', dtype=torch.float16)
a = torch.tensor([[0.1,0.2],[0.3,0.4]], device='cuda', dtype=torch.float16)
b = torch.tensor([[0.1,0.3],[0.2,0.4]], device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b.t())
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['K'],  # Argument names to use as an x-axis for the plot
        # x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        x_vals=[8192],
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['torch', 'triton'],
        # Label name for the lines
        line_names=["torch", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(K, provider):
    M = N = 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((N, K), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        # 对每个kernel进行25次的warm_up和100次iteration
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b.t()), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)