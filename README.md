# GPU Histogram Optimization

CUDA implementations of parallel histogram, progressing from naive global atomics to optimized shared memory with vectorized loads.

Companion code for [GPU Histogram: From Global Atomics to Shared Memory Privatization](https://yencal.github.io/gpu-histogram-optimization/).

## Results (H100)

**Configuration**
- Input: 1B unsigned char elements (uniform random)
- Bins: 256
- Block size: 512

| Algorithm | Time (ms) | Bandwidth | % Peak |
|-----------|-----------|-----------|--------|
| Global Atomic | 276.14 | 3.89 GB/s | 0.1% |
| Shared Atomic | 1.53 | 700.21 GB/s | 20.9% |
| Shared Atomic (vectorized) | 0.47 | 2297.21 GB/s | 68.5% |
| CUB DeviceHistogram | 0.50 | 2139.35 GB/s | 63.8% |


## Build & Run
```bash
nvcc -O3 -std=c++17 -arch=sm_90 main.cu -lcurand -o histogram_bench
./histogram_bench [power]  # default: 2^30 elements
```

## Files

- `histogram_kernels.cuh`: All histogram implementations (global atomic, shared atomic, vectorized)
- `utils.cuh`: Error checking, benchmark utilities, and CUB comparison
- `main.cu`: Benchmark runner