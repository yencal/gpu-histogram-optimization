#include "utils.cuh"
#include "histogram_kernels.cuh"

int main(int argc, char** argv)
{
    constexpr int BLOCK_SIZE = 512;
    constexpr int NUM_BINS = 256;

    // Default: 2^30 unsigned char elements
    int power = 30;

    if (argc >= 2) {
        power = std::atoi(argv[1]);
        if (power < 1 || power > 30) {
            std::cerr << "Power should be between 1 and 30" << std::endl;
            return EXIT_FAILURE;
        }
    }

    const int num_elements = 1 << power;

    std::cout << "========================================" << std::endl;
    std::cout << "GPU Histogram Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Array size: 2^" << power << " = " << num_elements << " unsigned char elements" << std::endl;
    std::cout << "Data size: " << (static_cast<size_t>(num_elements) * sizeof(unsigned char)) / (1024.0 * 1024.0 * 1024.0) 
              << " GB" << std::endl;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;
    std::cout << "Number of bins: " << NUM_BINS << std::endl;
    std::cout << "Device peak bandwidth: " << GetPeakBandwidth() << " GB/s" << std::endl;

    RunBenchmark<NUM_BINS, BLOCK_SIZE>("Global Atomic", histogram_global_atomic<NUM_BINS>, num_elements);
    RunBenchmark<NUM_BINS, BLOCK_SIZE>("Shared Atomic", histogram_shared_atomic<NUM_BINS>, num_elements);
    RunBenchmark<NUM_BINS, BLOCK_SIZE>("Warp Private", histogram_warp_private<NUM_BINS, BLOCK_SIZE>, num_elements);

    return EXIT_SUCCESS;
}