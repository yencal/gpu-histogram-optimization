// utils.cuh
// Error checking and benchmark utilities

#pragma once

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// CUDA ERROR CHECKING
// ============================================================================

#define CHECK_CUDA(val) CheckCuda((val), #val, __FILE__, __LINE__)

inline void CheckCuda(cudaError_t err, const char* const func, 
                      const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================================
// DEVICE INFO
// ============================================================================

inline float GetPeakBandwidth()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    int memory_clock_khz;
    int memory_bus_width_bits;
    
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, 
                                      cudaDevAttrMemoryClockRate, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_bus_width_bits, 
                                      cudaDevAttrGlobalMemoryBusWidth, device));
    
    // DDR: multiply by 2
    float peak_bandwidth_gbs = 2.0f * memory_clock_khz * 
                               (memory_bus_width_bits / 8.0f) / 1e6f;
    
    return peak_bandwidth_gbs;
}

inline int GetNumSMs()
{
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    int num_SMs = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&num_SMs,
                                      cudaDevAttrMultiProcessorCount, device));
    return num_SMs;
}


// ============================================================================
// TEST DATA INITIALIZATION AND VERIFICATION
// ============================================================================

inline void InitializeTestData(unsigned char* data, int num_elements, int* histogram, int num_bins)
{
    for (int i = 0; i < num_elements; ++i) {
        data[i] = rand() % num_bins;
    }

    for (int bin = 0; bin < num_bins; ++bin) {
        histogram[bin] = 0;
    }

    for (int i = 0; i < num_elements; ++i) {
        histogram[data[i]]++;
    }
}

inline bool VerifyHistogram(const int* output, const int* gold, int num_bins)
{
    for (int i = 0; i < n; ++i) {
        if (output[i] != gold[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << "got " << output[i] << ", expected " << gold[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// BENCHMARK RUNNER
// ============================================================================

template<typename HistogramKernel>
void RunBenchmark(
    const char* label,
    int num_elements,
    int num_bins,
    int block_size,
    int warmup_runs = 2,
    int timed_runs = 10)
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "Testing: " << label << std::endl;
    std::cout << "================================================" << std::endl;


    size_t data_bytes = num_elements * sizeof(unsigned char);
    size_t histogram_bytes = num_bins * sizeof(int);

    // Allocate and initialize host memory
    unsigned char* h_data = new unsigned char[num_elements];
    int* h_gold = new int[num_bins];
    InitializeTestData(h_data, num_elements, h_gold, num_bins);

    // Allocate device memory
    unsigned char* d_data;
    int* d_histogram;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMalloc(&d_histogram, histogram_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int max_grid_size = GetNumSMs() * 32;
    int grid_size = (num_elements + block_size - 1)/ block_size;
    grid_size = std::min(grid_size, max_grid_size);

    // Warmup runs
    for (int i = 0; i < warmup_runs; ++i) {
        cudaMemset(d_histogram, 0, histogram_bytes);
        HistogramKernel<<<grid_size, block_size>>>(d_data, num_elements, d_histogram, num_bins);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Verify correctness (fail fast)
    int* h_histogram = new int[num_bins];
    CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, histogram_bytes, cudaMemcpyDeviceToHost));

    if (!VerifyHistogram(h_histogram, h_gold, num_bins)) {
        std::cout << "FAILED: " << label << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::cout << "Correctness: PASSED" << std::endl;

    // Timed runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times;
    for (int i = 0; i < timed_runs; ++i) {
        cudaMemset(d_histogram, 0, histogram_bytes);

        CHECK_CUDA(cudaEventRecord(start));
        HistogramKernel<<<grid_size, block_size>>>(d_data, num_elements, d_histogram, num_bins);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float time_ms;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    // Calculate statistics
    float sum_time = 0.0f;
    float min_time = times[0];
    float max_time = times[0];
    for (const float t : times) {
        sum_time += t;
        min_time = std::min(t, min_time);
        max_time = std::max(t, max_time);
    }
    float avg_time = sum_time / timed_runs;

    // Calculate bandwidth (read only. write is negligible)
    float bandwidth_gbs = (data_bytes / 1e9) / (avg_time / 1000.0f);
    float percent_peak = (bandwidth_gbs / GetPeakBandwidth()) * 100.0f;

    // Print results
    std::cout << "Time (avg/min/max): " 
              << avg_time << " / " << min_time << " / " << max_time << " ms" << std::endl;
    std::cout << "Bandwidth: " << bandwidth_gbs << " GB/s (" 
              << percent_peak << "% of peak)" << std::endl;
    
    // Cleanup
    delete[] h_data;
    delete[] h_gold;
    delete[] h_histogram;
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}