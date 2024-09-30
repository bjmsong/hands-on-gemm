import itertools
import torch
import triton
import triton.language as tl
from triton import Config
import sys
sys.path.append(".")
from utils.quantize_rowwise import quantize_rowwise

def alt_wo_mm(x, w_int8, scale):
    return (x.unsqueeze(-1) * w_int8.unsqueeze(0)).sum(dim=1) * scale
    # return torch.mm(x, w_int8)*scale+bias

D = 2**6
N = D
x = torch.randn((1, D), device = 'cuda', dtype=torch.bfloat16)
w = torch.randn((N, D), device='cuda', dtype=torch.bfloat16)

w_int8, scale = quantize_rowwise(w)
scale = torch.randn(N, device = 'cuda', dtype=torch.bfloat16)

comp_fn = torch.compile(alt_wo_mm, mode='max-autotune')
triton_output = comp_fn(x, w_int8.t(), scale)
torch_output = torch.matmul(x, w.t())
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")