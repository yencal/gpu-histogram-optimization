// histogram_kernels.cuh

template<int NUM_BINS>
__global__ void HistogramGlobalAtomic(
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
__global__ void HistogramSharedAtomic(
    const unsigned char* __restrict__ data,
    int num_elements,
    int* __restrict__ histogram)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Initialize block-private histogram
    __shared__ int histogram_s[NUM_BINS];
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histogram_s[bin] = 0;
    }
    __syncthreads();

    // Accumulate into shared memory
    for (int i = idx; i < num_elements; i += stride) {
        atomicAdd(&histogram_s[data[i]], 1);
    }
    __syncthreads();

    // Write block histogram to global
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int bin_value = histogram_s[bin];
        if (bin_value > 0) {
            atomicAdd(&histogram[bin], bin_value);
        }
    }
}

template<int NUM_BINS>
__global__ void HistogramSharedAtomicVec(
    const unsigned char* __restrict__ data,
    int num_elements,
    int* __restrict__ histogram)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Initialize block-private histogram
    __shared__ int histogram_s[NUM_BINS];
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        histogram_s[bin] = 0;
    }
    __syncthreads();

    // Vectorized accumulation (16 bytes per load)
    constexpr int VEC_SIZE = sizeof(uint4);
    unsigned char values[VEC_SIZE];
    const int num_vecs = num_elements * sizeof(unsigned char) / VEC_SIZE;
    const uint4* data_vec = reinterpret_cast<const uint4*>(data);
    for (int i = idx; i < num_vecs; i += stride) {
        *reinterpret_cast<uint4*>(values) = data_vec[i];
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
            atomicAdd(&histogram_s[values[j]], 1);
        }
    }

    // Handle tail elements
    const int tail_start = num_vecs * VEC_SIZE;
    for (int i = tail_start + idx; i < num_elements; i += stride) {
        atomicAdd(&histogram_s[data[i]], 1);
    }
    __syncthreads();

    // Write block histogram to global
    for (int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        int bin_value = histogram_s[bin];
        if (bin_value > 0) {
            atomicAdd(&histogram[bin], bin_value);
        }
    }
}