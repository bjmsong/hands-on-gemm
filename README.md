## Testbed
- RTX 4090
- CUDA 12.1
- CUTLASS 3.4.1
- cuBLAS 12.01
- Warm up : 100 times 
- Execution : 100 times 
- DataType: fp32 + fp16

## Performance
- M=K=N=8192
![](./performance/1.png)

- M=N=512，K=8192
![](./performance/2.png)