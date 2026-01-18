// histogram_kernels.cuh

template<int NUM_BINS>
__global__ void histogram_global_atomic(
    const unsigned char* __restrict__ data,
    int num_elements,
    int* __restrict__ histogram)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < num_elements; i += stride)
    {
        atomicAdd(&histogram[data[i]], 1);
    }
}

template<int NUM_BINS>
__global__ void histogram_shared_atomic(
    const unsigned char* __restrict__ data,
    int num_elements,
    int* __restrict__ histogram)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    __shared__ int histogram_s[NUM_BINS];
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histogram_s[bin] = 0u;
    }
    __syncthreads();

    for (int i = tid; i < num_elements; i += stride) {
        atomicAdd(&histogram_s[data[i]], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int bin_value = histogram_s[bin];
        if (bin_value > 0) {
            atomicAdd(&histogram[bin], bin_value);
        }
    }
}

template<int NUM_BINS, int BLOCK_SIZE>
__global__ void histogram_warp_private(
    const unsigned char* __restrict__ data,
    int num_elements,
    int* __restrict__ histogram)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int warp_idx = threadIdx.x / warpSize;
    const int lane = threadIdx.x % warpSize;
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    __shared__ int histogram_s[NUM_WARPS][NUM_BINS];
    for (int bin = lane; bin < NUM_BINS; bin += warpSize) {
        histogram_s[warp_idx][bin] = 0u;
    }
    __syncwarp();

    for (int i = idx; i < num_elements; i += stride) {
        atomicAdd(&histogram_s[warp_idx][data[i]], 1);
    }
    __syncwarp();

    for (int bin = lane; bin < NUM_BINS; bin += warpSize) {
        int bin_value = histogram_s[warp_idx][bin];
        if (bin_value > 0) {
            atomicAdd(&histogram[bin], bin_value);
        }
    }
}