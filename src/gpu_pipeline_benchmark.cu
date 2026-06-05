#include "gpu_pipeline.h"

#include <cuda_runtime.h>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

#include "kernels.cuh"
#include "utils.h"

namespace {

constexpr int kWarmupRuns = 3;      // 预热运行次数
constexpr int kBenchmarkRuns = 10;  // 基准测试运行次数

/**
 * @brief 运行单次GPU流水线（可选择使用哪些optimized kernel）
 * @param input_image 输入图像数据
 * @param use_optimized_rgb 是否使用optimized RGB转灰度kernel
 * @param use_optimized_blur 是否使用optimized高斯模糊kernel
 * @param use_optimized_histogram 是否使用optimized直方图kernel
 * @param use_optimized_sobel 是否使用optimized Sobel kernel
 * @return 核函数执行时间（毫秒）
 */
float run_single_pipeline(const ImageData& input_image,
                          bool use_optimized_rgb,
                          bool use_optimized_blur,
                          bool use_optimized_histogram,
                          bool use_optimized_sobel) {
    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_equalized = nullptr;
    unsigned char* d_edge = nullptr;
    unsigned int* d_histogram = nullptr;
    unsigned char* d_lut = nullptr;

    cudaEvent_t start_kernel = nullptr;
    cudaEvent_t stop_kernel = nullptr;

    const auto cleanup = [&]() {
        if (stop_kernel != nullptr) cudaEventDestroy(stop_kernel);
        if (start_kernel != nullptr) cudaEventDestroy(start_kernel);
        if (d_lut != nullptr) cudaFree(d_lut);
        if (d_histogram != nullptr) cudaFree(d_histogram);
        if (d_edge != nullptr) cudaFree(d_edge);
        if (d_equalized != nullptr) cudaFree(d_equalized);
        if (d_blur != nullptr) cudaFree(d_blur);
        if (d_gray != nullptr) cudaFree(d_gray);
        if (d_rgb != nullptr) cudaFree(d_rgb);
    };

    try {
        const int width = input_image.width;
        const int height = input_image.height;
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const size_t rgb_bytes = gray_bytes * 3;

        if (input_image.channels == 3) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_rgb, rgb_bytes));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_gray, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_blur, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_equalized, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_edge, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, kHistogramBins * sizeof(unsigned int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_lut, kHistogramBins * sizeof(unsigned char)));

        CHECK_CUDA_ERROR(cudaEventCreate(&start_kernel));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_kernel));

        // 传输输入数据到GPU
        if (input_image.channels == 3) {
            CHECK_CUDA_ERROR(cudaMemcpy(d_rgb,
                                       input_image.pixels.data(),
                                       rgb_bytes,
                                       cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA_ERROR(cudaMemcpy(d_gray,
                                       input_image.pixels.data(),
                                       gray_bytes,
                                       cudaMemcpyHostToDevice));
        }

        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));
        const int threads = 256;
        const int linear_blocks = div_up(pixels, threads);

        CHECK_CUDA_ERROR(cudaEventRecord(start_kernel));

        // 阶段1: RGB转灰度
        if (input_image.channels == 3) {
            if (use_optimized_rgb) {
                rgb_to_gray_optimized_kernel<<<grid, block>>>(
                    reinterpret_cast<const uchar3*>(d_rgb), d_gray, width, height);
            } else {
                rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
            }
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        // 阶段2: 高斯模糊
        if (use_optimized_blur) {
            gaussian_blur_optimized_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
        } else {
            gaussian_blur_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 阶段3: 直方图均衡化
        CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
        if (use_optimized_histogram) {
            compute_histogram_optimized_kernel<<<linear_blocks, threads>>>(d_blur, d_histogram, pixels);
        } else {
            compute_histogram_global_kernel<<<linear_blocks, threads>>>(d_blur, d_histogram, pixels);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());

        build_equalization_lut_kernel<<<1, kHistogramBins>>>(d_histogram, d_lut, pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());

        apply_lut_kernel<<<linear_blocks, threads>>>(d_blur, d_equalized, d_lut, pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 阶段4: Sobel边缘检测
        if (use_optimized_sobel) {
            sobel_edge_optimized_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
        } else {
            sobel_edge_naive_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
        }
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_kernel));

        float kernel_time_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel));

        cleanup();
        return kernel_time_ms;
    } catch (...) {
        cleanup();
        throw;
    }
}

/**
 * @brief 统计并打印基准测试结果
 * @param results 多次运行的时间结果
 * @param test_name 测试名称
 */
void print_benchmark_result(const std::vector<float>& results, const std::string& test_name) {
    // 跳过预热运行
    std::vector<float> valid_results(results.begin() + kWarmupRuns, results.end());

    // 计算统计信息
    float sum = std::accumulate(valid_results.begin(), valid_results.end(), 0.0f);
    float avg = sum / valid_results.size();

    // 计算最小值和最大值
    float min_val = *std::min_element(valid_results.begin(), valid_results.end());
    float max_val = *std::max_element(valid_results.begin(), valid_results.end());

    // 计算标准差
    float variance = 0.0f;
    for (float val : valid_results) {
        variance += (val - avg) * (val - avg);
    }
    variance /= valid_results.size();
    float stddev = std::sqrt(variance);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << test_name << ":" << std::endl;
    std::cout << "  Runs:      " << valid_results.size() << std::endl;
    std::cout << "  Average:   " << avg << " ms" << std::endl;
    std::cout << "  Min:       " << min_val << " ms" << std::endl;
    std::cout << "  Max:       " << max_val << " ms" << std::endl;
    std::cout << "  StdDev:    " << stddev << " ms" << std::endl;
}

}  // namespace

void run_gpu_benchmark(const ImageData& input_image) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   GPU Kernel Performance Benchmark" << std::endl;
    std::cout << "========================================\n" << std::endl;

    print_gpu_device_info();

    const int width = input_image.width;
    const int height = input_image.height;
    const int pixels = width * height;
    std::cout << "Image size: " << width << " x " << height
              << " (" << pixels << " pixels)" << std::endl;
    std::cout << "Warmup runs: " << kWarmupRuns << std::endl;
    std::cout << "Benchmark runs: " << kBenchmarkRuns << std::endl;

    // 测试配置：每轮只用一个optimized kernel
    struct TestConfig {
        std::string name;
        bool opt_rgb;
        bool opt_blur;
        bool opt_histogram;
        bool opt_sobel;
    };

    std::vector<TestConfig> tests = {
        {"Baseline (All Naive)",              false, false, false, false},
        {"Only RGB Optimized",                true,  false, false, false},
        {"Only Gaussian Blur Optimized",      false, true,  false, false},
        {"Only Histogram Optimized",          false, false, true,  false},
        {"Only Sobel Optimized",              false, false, false, true},
        {"All Optimized",                     true,  true,  true,  true}
    };

    // 存储所有测试结果
    std::vector<std::vector<float>> all_results(tests.size());

    // 运行所有测试
    for (size_t test_idx = 0; test_idx < tests.size(); ++test_idx) {
        const auto& test = tests[test_idx];
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Running: " << test.name << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        auto& results = all_results[test_idx];
        results.resize(kWarmupRuns + kBenchmarkRuns);

        for (int run = 0; run < kWarmupRuns + kBenchmarkRuns; ++run) {
            float time = run_single_pipeline(input_image,
                                            test.opt_rgb,
                                            test.opt_blur,
                                            test.opt_histogram,
                                            test.opt_sobel);
            results[run] = time;

            if (run < kWarmupRuns) {
                std::cout << "  Warmup " << (run + 1) << ": " << time << " ms" << std::endl;
            } else {
                std::cout << "  Run " << (run - kWarmupRuns + 1) << ": " << time << " ms" << std::endl;
            }
        }
    }

    // 打印统计结果
    std::cout << "\n========================================" << std::endl;
    std::cout << "         Benchmark Results Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    for (size_t test_idx = 0; test_idx < tests.size(); ++test_idx) {
        print_benchmark_result(all_results[test_idx], tests[test_idx].name);
    }

    // 计算性能提升
    std::cout << "\n========================================" << std::endl;
    std::cout << "         Performance Improvements" << std::endl;
    std::cout << "========================================\n" << std::endl;

    float baseline_avg = 0.0f;
    {
        const auto& results = all_results[0];
        std::vector<float> valid(results.begin() + kWarmupRuns, results.end());
        baseline_avg = std::accumulate(valid.begin(), valid.end(), 0.0f) / valid.size();
    }

    std::cout << std::fixed << std::setprecision(2);
    for (size_t test_idx = 1; test_idx < tests.size(); ++test_idx) {
        const auto& results = all_results[test_idx];
        std::vector<float> valid(results.begin() + kWarmupRuns, results.end());
        float avg = std::accumulate(valid.begin(), valid.end(), 0.0f) / valid.size();

        float improvement = ((baseline_avg - avg) / baseline_avg) * 100.0f;
        std::cout << tests[test_idx].name << ":" << std::endl;
        std::cout << "  Time: " << avg << " ms (vs baseline " << baseline_avg << " ms)" << std::endl;
        std::cout << "  Improvement: " << improvement << "%" << std::endl;
        std::cout << std::endl;
    }
}
