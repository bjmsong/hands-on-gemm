import itertools
import torch
import triton
import triton.language as tl
from triton import Config
import sys
sys.path.append(".")
from utils.quantize_rowwise import quantize_rowwise

def alt_wo_mm(x, w_int8, scale):
    # return (x.unsqueeze(-1) * w_int8.unsqueeze(0)).sum(dim=1) * scale   # 逐元素相乘
    return torch.mm(x, w_int8.to(torch.bfloat16))*scale/127.0

torch.manual_seed(0)
M, K, N = 32, 1024, 1024
# M, K, N = 16384, 16384, 16384
x = torch.randn((M, K), device = 'cuda', dtype=torch.bfloat16)
w = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
w_int8, scale = quantize_rowwise(w)

comp_fn = torch.compile(alt_wo_mm, mode='max-autotune')
triton_output = comp_fn(x, w_int8.t(), scale)
torch_output = torch.matmul(x, w.t())
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")


M_range = [2 ** i for i in range(0, 15, 2)]
N_K_range = [2 ** i for i in range(10, 15, 2)]
matrix_range = list(itertools.product(M_range, N_K_range, N_K_range))
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[list(_) for _ in matrix_range],  # Different possible values for `x_name`
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
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    W = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
    W_int8, scale = quantize_rowwise(W)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        # 对每个kernel进行25次的warm_up和100次iteration
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, W.t()), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: comp_fn(a, W_int8.t(), scale), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-9 / ms
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True, save_path="plot/")