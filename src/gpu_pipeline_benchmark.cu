#include "gpu_pipeline.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <string>
#include <vector>

#include "kernels.cuh"
#include "utils.h"

namespace {

constexpr int kWarmupRuns = 3;
constexpr int kBenchmarkRuns = 10;

struct PreparedInputs {
    std::vector<unsigned char> gray;
    std::vector<unsigned char> blur;
    std::vector<unsigned char> equalized;
};

struct VariantResults {
    std::vector<float> naive;
    std::vector<float> optimized;
};

struct Stats {
    float average = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    float stddev = 0.0f;
};

void rgb_to_gray_reference(const ImageData& input, std::vector<unsigned char>& gray) {
    const int pixels = input.width * input.height;
    if (input.channels == 1) {
        std::copy(input.pixels.begin(), input.pixels.end(), gray.begin());
        return;
    }

    for (int i = 0; i < pixels; ++i) {
        const int base = i * 3;
        const float r = static_cast<float>(input.pixels[base + 0]);
        const float g = static_cast<float>(input.pixels[base + 1]);
        const float b = static_cast<float>(input.pixels[base + 2]);
        const float gray_value = 0.299f * r + 0.587f * g + 0.114f * b;
        gray[i] = static_cast<unsigned char>(gray_value);
    }
}

void gaussian_blur_reference(const std::vector<unsigned char>& input,
                             std::vector<unsigned char>& output,
                             int width,
                             int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int sum = 0;
            for (int ky = -kGaussianRadius; ky <= kGaussianRadius; ++ky) {
                const int yy = clamp_int(y + ky, 0, height - 1);
                for (int kx = -kGaussianRadius; kx <= kGaussianRadius; ++kx) {
                    const int xx = clamp_int(x + kx, 0, width - 1);
                    const int kernel_index = (ky + kGaussianRadius) * kGaussianWidth +
                                             (kx + kGaussianRadius);
                    sum += static_cast<int>(input[yy * width + xx]) *
                           kGaussian5x5[kernel_index];
                }
            }
            output[y * width + x] = static_cast<unsigned char>(
                (sum + kGaussianWeightSum / 2) / kGaussianWeightSum);
        }
    }
}

void histogram_equalization_reference(const std::vector<unsigned char>& input,
                                      std::vector<unsigned char>& output) {
    unsigned int histogram[kHistogramBins] = {};
    unsigned int cdf[kHistogramBins] = {};
    unsigned char lut[kHistogramBins] = {};

    for (unsigned char pixel : input) {
        ++histogram[pixel];
    }

    cdf[0] = histogram[0];
    for (int i = 1; i < kHistogramBins; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    unsigned int cdf_min = 0;
    for (int i = 0; i < kHistogramBins; ++i) {
        if (cdf[i] != 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    const int pixels = static_cast<int>(input.size());
    if (pixels <= static_cast<int>(cdf_min)) {
        for (int i = 0; i < kHistogramBins; ++i) {
            lut[i] = static_cast<unsigned char>(i);
        }
    } else {
        for (int i = 0; i < kHistogramBins; ++i) {
            const int numerator =
                (static_cast<int>(cdf[i]) - static_cast<int>(cdf_min)) * 255;
            const int denominator = pixels - static_cast<int>(cdf_min);
            lut[i] = static_cast<unsigned char>(clamp_int(numerator / denominator, 0, 255));
        }
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = lut[input[i]];
    }
}

PreparedInputs prepare_inputs(const ImageData& input_image) {
    const int pixels = input_image.width * input_image.height;
    PreparedInputs prepared;
    prepared.gray.resize(pixels);
    prepared.blur.resize(pixels);
    prepared.equalized.resize(pixels);

    rgb_to_gray_reference(input_image, prepared.gray);
    gaussian_blur_reference(prepared.gray,
                            prepared.blur,
                            input_image.width,
                            input_image.height);
    histogram_equalization_reference(prepared.blur, prepared.equalized);
    return prepared;
}

template <typename LaunchFn>
float time_kernel(cudaEvent_t start, cudaEvent_t stop, LaunchFn&& launch) {
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    launch();
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

template <typename RunFn>
std::vector<float> run_series(const std::string& label, RunFn&& run_once) {
    std::cout << "\n  " << label << std::endl;

    std::vector<float> results(kWarmupRuns + kBenchmarkRuns);
    for (int run = 0; run < kWarmupRuns + kBenchmarkRuns; ++run) {
        results[run] = run_once();
        if (run < kWarmupRuns) {
            std::cout << "    Warmup " << (run + 1) << ": " << results[run] << " ms"
                      << std::endl;
        } else {
            std::cout << "    Run " << (run - kWarmupRuns + 1) << ": " << results[run]
                      << " ms" << std::endl;
        }
    }
    return results;
}

Stats compute_stats(const std::vector<float>& results) {
    const auto valid_begin = results.begin() + kWarmupRuns;
    const auto valid_end = results.end();
    const float sum = std::accumulate(valid_begin, valid_end, 0.0f);
    const float average = sum / static_cast<float>(kBenchmarkRuns);
    const float min_value = *std::min_element(valid_begin, valid_end);
    const float max_value = *std::max_element(valid_begin, valid_end);

    float variance = 0.0f;
    for (auto it = valid_begin; it != valid_end; ++it) {
        const float delta = *it - average;
        variance += delta * delta;
    }
    variance /= static_cast<float>(kBenchmarkRuns);

    return Stats{average, min_value, max_value, std::sqrt(variance)};
}

float average_of(const std::vector<float>& results) {
    return compute_stats(results).average;
}

void print_comparison_summary(const std::string& test_name,
                              const VariantResults& results,
                              const std::string& measurement_note = "") {
    const Stats naive = compute_stats(results.naive);
    const Stats optimized = compute_stats(results.optimized);
    const float improvement =
        naive.average > 0.0f ? ((naive.average - optimized.average) / naive.average) * 100.0f
                             : 0.0f;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n" << test_name << ":" << std::endl;
    if (!measurement_note.empty()) {
        std::cout << "  Note:      " << measurement_note << std::endl;
    }
    std::cout << "  Naive Avg: " << naive.average << " ms"
              << "  (min " << naive.min << ", max " << naive.max
              << ", stddev " << naive.stddev << ")" << std::endl;
    std::cout << "  Opt Avg:   " << optimized.average << " ms"
              << "  (min " << optimized.min << ", max " << optimized.max
              << ", stddev " << optimized.stddev << ")" << std::endl;
    std::cout << std::setprecision(2);
    std::cout << "  Improvement: " << improvement << "%" << std::endl;
}


VariantResults benchmark_rgb_kernel(const ImageData& input_image) {
    VariantResults results;
    if (input_image.channels != 3) {
        return results;
    }

    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        if (d_gray != nullptr) cudaFree(d_gray);
        if (d_rgb != nullptr) cudaFree(d_rgb);
    };

    try {
        const int width = input_image.width;
        const int height = input_image.height;
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const size_t rgb_bytes = gray_bytes * 3;
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));

        CHECK_CUDA_ERROR(cudaMalloc(&d_rgb, rgb_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_gray, gray_bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_rgb,
                                    input_image.pixels.data(),
                                    rgb_bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        results.naive = run_series("Naive RGB -> Gray", [&]() {
            return time_kernel(start, stop, [&]() {
                rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        results.optimized = run_series("Optimized RGB -> Gray", [&]() {
            return time_kernel(start, stop, [&]() {
                rgb_to_gray_optimized_kernel<<<grid, block>>>(
                    reinterpret_cast<const uchar3*>(d_rgb), d_gray, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        cleanup();
        return results;
    } catch (...) {
        cleanup();
        throw;
    }
}

VariantResults benchmark_blur_kernel(const PreparedInputs& prepared,
                                     int width,
                                     int height) {
    VariantResults results;
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        if (d_output != nullptr) cudaFree(d_output);
        if (d_input != nullptr) cudaFree(d_input);
    };

    try {
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, gray_bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                    prepared.gray.data(),
                                    gray_bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        results.naive = run_series("Naive Gaussian Blur", [&]() {
            return time_kernel(start, stop, [&]() {
                gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        results.optimized = run_series("Optimized Gaussian Blur", [&]() {
            return time_kernel(start, stop, [&]() {
                gaussian_blur_optimized_kernel<<<grid, block>>>(d_input,
                                                                d_output,
                                                                width,
                                                                height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        cleanup();
        return results;
    } catch (...) {
        cleanup();
        throw;
    }
}

VariantResults benchmark_histogram_kernel(const PreparedInputs& prepared, int pixels) {
    VariantResults results;
    unsigned char* d_input = nullptr;
    unsigned int* d_histogram = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        if (d_histogram != nullptr) cudaFree(d_histogram);
        if (d_input != nullptr) cudaFree(d_input);
    };

    try {
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const int threads = 256;
        const int linear_blocks = div_up(pixels, threads);

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, kHistogramBins * sizeof(unsigned int)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                    prepared.blur.data(),
                                    gray_bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        results.naive = run_series("Naive Histogram Kernel", [&]() {
            CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            return time_kernel(start, stop, [&]() {
                compute_histogram_global_kernel<<<linear_blocks, threads>>>(
                    d_input, d_histogram, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        results.optimized = run_series("Optimized Histogram Kernel", [&]() {
            CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            return time_kernel(start, stop, [&]() {
                compute_histogram_optimized_kernel<<<linear_blocks, threads>>>(
                    d_input, d_histogram, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        cleanup();
        return results;
    } catch (...) {
        cleanup();
        throw;
    }
}

VariantResults benchmark_sobel_kernel(const PreparedInputs& prepared,
                                      int width,
                                      int height) {
    VariantResults results;
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        if (d_output != nullptr) cudaFree(d_output);
        if (d_input != nullptr) cudaFree(d_input);
    };

    try {
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));

        CHECK_CUDA_ERROR(cudaMalloc(&d_input, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_output, gray_bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_input,
                                    prepared.equalized.data(),
                                    gray_bytes,
                                    cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        results.naive = run_series("Naive Sobel", [&]() {
            return time_kernel(start, stop, [&]() {
                sobel_edge_naive_kernel<<<grid, block>>>(d_input, d_output, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        results.optimized = run_series("Optimized Sobel", [&]() {
            return time_kernel(start, stop, [&]() {
                sobel_edge_optimized_kernel<<<grid, block>>>(d_input, d_output, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        cleanup();
        return results;
    } catch (...) {
        cleanup();
        throw;
    }
}

VariantResults benchmark_full_pipeline(const ImageData& input_image) {
    VariantResults results;
    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_equalized = nullptr;
    unsigned char* d_edge = nullptr;
    unsigned int* d_histogram = nullptr;
    unsigned char* d_lut = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    const auto cleanup = [&]() {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
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
        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));
        const int threads = 256;
        const int linear_blocks = div_up(pixels, threads);

        if (input_image.channels == 3) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_rgb, rgb_bytes));
            CHECK_CUDA_ERROR(cudaMemcpy(d_rgb,
                                        input_image.pixels.data(),
                                        rgb_bytes,
                                        cudaMemcpyHostToDevice));
        } else {
            CHECK_CUDA_ERROR(cudaMalloc(&d_gray, gray_bytes));
            CHECK_CUDA_ERROR(cudaMemcpy(d_gray,
                                        input_image.pixels.data(),
                                        gray_bytes,
                                        cudaMemcpyHostToDevice));
        }

        if (d_gray == nullptr) {
            CHECK_CUDA_ERROR(cudaMalloc(&d_gray, gray_bytes));
        }
        CHECK_CUDA_ERROR(cudaMalloc(&d_blur, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_equalized, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_edge, gray_bytes));
        CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, kHistogramBins * sizeof(unsigned int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_lut, kHistogramBins * sizeof(unsigned char)));
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));

        results.naive = run_series("Naive Full Pipeline", [&]() {
            return time_kernel(start, stop, [&]() {
                if (input_image.channels == 3) {
                    rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                }

                gaussian_blur_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());

                CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
                compute_histogram_global_kernel<<<linear_blocks, threads>>>(
                    d_blur, d_histogram, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                build_equalization_lut_kernel<<<1, kHistogramBins>>>(d_histogram, d_lut, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                apply_lut_kernel<<<linear_blocks, threads>>>(d_blur, d_equalized, d_lut, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                sobel_edge_naive_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        results.optimized = run_series("Optimized Full Pipeline", [&]() {
            return time_kernel(start, stop, [&]() {
                if (input_image.channels == 3) {
                    rgb_to_gray_optimized_kernel<<<grid, block>>>(
                        reinterpret_cast<const uchar3*>(d_rgb), d_gray, width, height);
                    CHECK_CUDA_ERROR(cudaGetLastError());
                }

                gaussian_blur_optimized_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());

                CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
                compute_histogram_optimized_kernel<<<linear_blocks, threads>>>(
                    d_blur, d_histogram, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                build_equalization_lut_kernel<<<1, kHistogramBins>>>(d_histogram, d_lut, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                apply_lut_kernel<<<linear_blocks, threads>>>(d_blur, d_equalized, d_lut, pixels);
                CHECK_CUDA_ERROR(cudaGetLastError());

                sobel_edge_optimized_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
                CHECK_CUDA_ERROR(cudaGetLastError());
            });
        });

        cleanup();
        return results;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace

void run_gpu_benchmark(const ImageData& input_image) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "   GPU Kernel Microbenchmark" << std::endl;
    std::cout << "========================================\n" << std::endl;

    print_gpu_device_info();

    const int width = input_image.width;
    const int height = input_image.height;
    const int pixels = width * height;
    std::cout << "Image size: " << width << " x " << height
              << " (" << pixels << " pixels)" << std::endl;
    std::cout << "Warmup runs: " << kWarmupRuns << std::endl;
    std::cout << "Benchmark runs: " << kBenchmarkRuns << std::endl;
    std::cout << "Timing scope: per-kernel timings exclude allocation, H2D/D2H copies, "
                 "and CPU-side input preparation." << std::endl;
    std::cout << "Histogram note: histogram reset is excluded; only the accumulation "
                 "kernel itself is timed." << std::endl;

    const PreparedInputs prepared = prepare_inputs(input_image);

    VariantResults rgb_results;
    if (input_image.channels == 3) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "RGB to Gray Kernel" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        rgb_results = benchmark_rgb_kernel(input_image);
    }

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Gaussian Blur Kernel" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    const VariantResults blur_results = benchmark_blur_kernel(prepared, width, height);

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Histogram Kernel" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    const VariantResults histogram_results = benchmark_histogram_kernel(prepared, pixels);

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Sobel Kernel" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    const VariantResults sobel_results = benchmark_sobel_kernel(prepared, width, height);

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Full Pipeline Total" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    const VariantResults pipeline_results = benchmark_full_pipeline(input_image);

    std::cout << "\n========================================" << std::endl;
    std::cout << "      Microbenchmark Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    if (input_image.channels == 3) {
        print_comparison_summary("RGB to Gray Kernel", rgb_results);
    }
    print_comparison_summary("Gaussian Blur Kernel", blur_results);
    print_comparison_summary("Histogram Kernel",
                             histogram_results,
                             "Only compute_histogram_*_kernel is timed.");
    print_comparison_summary("Sobel Kernel", sobel_results);
    print_comparison_summary("Full Pipeline Total",
                             pipeline_results,
                             "Includes cudaMemset + all pipeline kernels, but excludes "
                             "allocation and host-device transfer.");

    std::cout << "\n========================================" << std::endl;
    std::cout << "     Pipeline vs Standalone Check" << std::endl;
    std::cout << "========================================" << std::endl;

    if (input_image.channels == 3) {
        std::cout << std::fixed << std::setprecision(4)
                  << "Standalone RGB optimized avg:        "
                  << average_of(rgb_results.optimized) << " ms" << std::endl;
    }
    std::cout << std::fixed << std::setprecision(4)
              << "Standalone blur optimized avg:       "
              << average_of(blur_results.optimized) << " ms" << std::endl;
    std::cout << "Standalone histogram optimized avg:  "
              << average_of(histogram_results.optimized) << " ms" << std::endl;
    std::cout << "Standalone sobel optimized avg:      "
              << average_of(sobel_results.optimized) << " ms" << std::endl;
    std::cout << "Full pipeline optimized avg:         "
              << average_of(pipeline_results.optimized) << " ms" << std::endl;
}
