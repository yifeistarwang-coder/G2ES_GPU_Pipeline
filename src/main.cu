/**
 * G2ES image processing pipeline
 *
 * Pipeline: RGB -> grayscale -> Gaussian blur -> histogram equalization ->
 * Sobel edge detection.
 *
 * Usage:
 *   ./image_pipeline [--gpu|--cpu|--both] [input_image] [output_prefix]
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "kernels.h"

namespace {

constexpr int kBlockSize = 16;
constexpr int kGaussianRadius = 2;
constexpr int kGaussianWidth = 2 * kGaussianRadius + 1;
constexpr int kGaussianWeightSum = 159;
constexpr const char* kDefaultInputPath = "image/test_image.png";

const int kGaussian5x5[kGaussianWidth * kGaussianWidth] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

enum class RunMode {
    kGpu,
    kCpu,
    kBoth
};

struct Options {
    RunMode mode = RunMode::kGpu;
    std::string input_path = kDefaultInputPath;
    std::string output_prefix;
    bool output_prefix_provided = false;
    bool show_help = false;
};

struct ImageData {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> pixels;
};

struct PipelineOutputs {
    std::vector<unsigned char> gray;
    std::vector<unsigned char> blur;
    std::vector<unsigned char> equalized;
    std::vector<unsigned char> edge;
};

struct CpuTiming {
    double compute_total_ms = 0.0;
    double write_ms = 0.0;
};

struct GpuTiming {
    float kernel_total_ms = 0.0f;
    float transfer_and_kernel_ms = 0.0f;
    double write_ms = 0.0;
};

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t result,
                const char* expression,
                const char* file,
                int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result)
              << " (" << cudaGetErrorName(result) << ") in " << expression
              << std::endl;
    throw std::runtime_error("CUDA call failed.");
}

#define MAIN_CUDA_CHECK(val) check_cuda((val), #val, __FILE__, __LINE__)

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int div_up(const int value, const int divisor) {
    return (value + divisor - 1) / divisor;
}

int clamp_int(const int value, const int low, const int high) {
    return std::max(low, std::min(value, high));
}

std::string strip_extension(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path;
    }
    return path.substr(0, dot);
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " [--gpu|--cpu|--both] [input_image] [output_prefix]" << std::endl;
    std::cerr << "Default mode:  --gpu" << std::endl;
    std::cerr << "Default input: " << kDefaultInputPath << std::endl;
}

bool parse_mode_flag(const std::string& value, RunMode* mode) {
    if (value == "--gpu") {
        *mode = RunMode::kGpu;
        return true;
    }
    if (value == "--cpu") {
        *mode = RunMode::kCpu;
        return true;
    }
    if (value == "--both") {
        *mode = RunMode::kBoth;
        return true;
    }
    return false;
}

Options parse_options(int argc, char** argv) {
    Options options;
    int arg_index = 1;

    if (argc > 1) {
        const std::string first_arg = argv[1];
        if (first_arg == "--help" || first_arg == "-h") {
            options.show_help = true;
            return options;
        }
        if (parse_mode_flag(first_arg, &options.mode)) {
            arg_index = 2;
        } else if (first_arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + first_arg);
        }
    }

    const int remaining_args = argc - arg_index;
    if (remaining_args > 2) {
        throw std::runtime_error("Too many arguments.");
    }

    if (remaining_args >= 1) {
        options.input_path = argv[arg_index];
    }

    options.output_prefix_provided = (remaining_args == 2);
    options.output_prefix = options.output_prefix_provided
                                ? argv[arg_index + 1]
                                : strip_extension(options.input_path);
    return options;
}

ImageData load_image(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        throw std::runtime_error("Failed to open input image: " + path);
    }
    if (raw.depth() != CV_8U) {
        throw std::runtime_error("Only 8-bit input images are supported.");
    }

    ImageData image;
    image.width = raw.cols;
    image.height = raw.rows;

    cv::Mat normalized;
    if (raw.channels() == 1) {
        normalized = raw;
        image.channels = 1;
    } else if (raw.channels() == 3) {
        cv::cvtColor(raw, normalized, cv::COLOR_BGR2RGB);
        image.channels = 3;
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, normalized, cv::COLOR_BGRA2RGB);
        image.channels = 3;
    } else {
        throw std::runtime_error("Only grayscale, RGB, or RGBA images are supported.");
    }

    if (!normalized.isContinuous()) {
        normalized = normalized.clone();
    }

    const size_t bytes = static_cast<size_t>(image.width) *
                         static_cast<size_t>(image.height) *
                         static_cast<size_t>(image.channels);
    image.pixels.assign(normalized.data, normalized.data + bytes);
    return image;
}

void write_png(const std::string& path,
               const std::vector<unsigned char>& pixels,
               const int width,
               const int height) {
    cv::Mat image(height, width, CV_8UC1, const_cast<unsigned char*>(pixels.data()));
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to write output image: " + path);
    }
}

double write_pipeline_outputs(const std::string& output_prefix,
                              const PipelineOutputs& outputs,
                              const int width,
                              const int height) {
    const auto write_start = Clock::now();
    write_png(output_prefix + "_gray.png", outputs.gray, width, height);
    write_png(output_prefix + "_blur.png", outputs.blur, width, height);
    write_png(output_prefix + "_equalized.png", outputs.equalized, width, height);
    write_png(output_prefix + "_edge.png", outputs.edge, width, height);
    return elapsed_ms(write_start, Clock::now());
}

PipelineOutputs make_outputs(const int pixels) {
    PipelineOutputs outputs;
    outputs.gray.resize(pixels);
    outputs.blur.resize(pixels);
    outputs.equalized.resize(pixels);
    outputs.edge.resize(pixels);
    return outputs;
}

void print_gpu_device_info() {
    int device_id = 0;
    MAIN_CUDA_CHECK(cudaGetDevice(&device_id));

    cudaDeviceProp prop;
    MAIN_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int clock_khz = 0;
    MAIN_CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id));

    int mem_clock_khz = 0;
    MAIN_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz,
                                           cudaDevAttrMemoryClockRate,
                                           device_id));

    int mem_bus_width = 0;
    MAIN_CUDA_CHECK(cudaDeviceGetAttribute(&mem_bus_width,
                                           cudaDevAttrGlobalMemoryBusWidth,
                                           device_id));

    std::cout << "=== GPU Device Info ===" << std::endl;
    std::cout << "Device:           " << prop.name << std::endl;
    std::cout << "Compute Cap:      " << prop.major << "." << prop.minor << std::endl;
    std::cout << "GPU Clock:        " << clock_khz / 1000 << " MHz" << std::endl;
    std::cout << "Memory Clock:     " << mem_clock_khz / 1000 << " MHz" << std::endl;
    std::cout << "Memory Bus Width: " << mem_bus_width << " bits" << std::endl;
    std::cout << "SM Count:         " << prop.multiProcessorCount << std::endl;
    std::cout << "Global Memory:    " << prop.totalGlobalMem / (1024 * 1024) << " MB"
              << std::endl;
    std::cout << "Shared Mem/Block: " << prop.sharedMemPerBlock / 1024 << " KB"
              << std::endl;
    std::cout << "========================\n" << std::endl;
}

void rgb_to_gray_cpu(const ImageData& input, std::vector<unsigned char>& gray) {
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

void gaussian_blur_cpu(const std::vector<unsigned char>& input,
                       std::vector<unsigned char>& output,
                       const int width,
                       const int height) {
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

void histogram_equalization_cpu(const std::vector<unsigned char>& input,
                                std::vector<unsigned char>& output) {
    unsigned int histogram[kHistogramBins] = {};
    unsigned int cdf[kHistogramBins] = {};
    unsigned char lut[kHistogramBins] = {};

    for (const unsigned char pixel : input) {
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
            const int mapped = numerator / denominator;
            lut[i] = static_cast<unsigned char>(clamp_int(mapped, 0, 255));
        }
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = lut[input[i]];
    }
}

void sobel_edge_cpu(const std::vector<unsigned char>& input,
                    std::vector<unsigned char>& output,
                    const int width,
                    const int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
                output[y * width + x] = 0;
                continue;
            }

            const int gx =
                -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] -
                2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] -
                input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

            const int gy =
                -input[(y - 1) * width + (x - 1)] -
                2 * input[(y - 1) * width + x] -
                input[(y - 1) * width + (x + 1)] +
                input[(y + 1) * width + (x - 1)] +
                2 * input[(y + 1) * width + x] +
                input[(y + 1) * width + (x + 1)];

            const int magnitude = static_cast<int>(
                std::sqrt(static_cast<float>(gx * gx + gy * gy)));
            output[y * width + x] = static_cast<unsigned char>(std::min(255, magnitude));
        }
    }
}

CpuTiming run_cpu_pipeline(const ImageData& input_image,
                           const std::string& output_prefix) {
    const int width = input_image.width;
    const int height = input_image.height;
    const int pixels = width * height;
    PipelineOutputs outputs = make_outputs(pixels);

    std::cout << "\nStarting G2ES CPU Image Pipeline benchmark..." << std::endl;

    const auto pipeline_start = Clock::now();

    const auto gray_start = Clock::now();
    rgb_to_gray_cpu(input_image, outputs.gray);
    const auto gray_end = Clock::now();

    const auto blur_start = Clock::now();
    gaussian_blur_cpu(outputs.gray, outputs.blur, width, height);
    const auto blur_end = Clock::now();

    const auto equalize_start = Clock::now();
    histogram_equalization_cpu(outputs.blur, outputs.equalized);
    const auto equalize_end = Clock::now();

    const auto sobel_start = Clock::now();
    sobel_edge_cpu(outputs.equalized, outputs.edge, width, height);
    const auto sobel_end = Clock::now();

    const auto pipeline_end = Clock::now();

    const double write_ms = write_pipeline_outputs(output_prefix, outputs, width, height);

    const double compute_total_ms = elapsed_ms(pipeline_start, pipeline_end);
    std::cout << "\n=== CPU Timing Results ===" << std::endl;
    std::cout << "CPU RGB to gray:       " << elapsed_ms(gray_start, gray_end)
              << " ms" << std::endl;
    std::cout << "CPU Gaussian blur:     " << elapsed_ms(blur_start, blur_end)
              << " ms" << std::endl;
    std::cout << "CPU equalization:      " << elapsed_ms(equalize_start, equalize_end)
              << " ms" << std::endl;
    std::cout << "CPU Sobel edge:        " << elapsed_ms(sobel_start, sobel_end)
              << " ms" << std::endl;
    std::cout << "CPU compute total:     " << compute_total_ms << " ms" << std::endl;
    std::cout << "CPU write outputs:     " << write_ms << " ms" << std::endl;
    std::cout << "==========================" << std::endl;
    std::cout << "CPU outputs written with prefix: " << output_prefix << std::endl;

    return CpuTiming{compute_total_ms, write_ms};
}

GpuTiming run_gpu_pipeline(const ImageData& input_image,
                           const std::string& output_prefix) {
    unsigned char* d_rgb = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_equalized = nullptr;
    unsigned char* d_edge = nullptr;
    unsigned int* d_histogram = nullptr;
    unsigned char* d_lut = nullptr;

    cudaEvent_t start_kernel = nullptr;
    cudaEvent_t stop_kernel = nullptr;
    cudaEvent_t start_e2e = nullptr;
    cudaEvent_t stop_e2e = nullptr;

    const auto cleanup = [&]() {
        if (stop_e2e != nullptr) {
            cudaEventDestroy(stop_e2e);
        }
        if (start_e2e != nullptr) {
            cudaEventDestroy(start_e2e);
        }
        if (stop_kernel != nullptr) {
            cudaEventDestroy(stop_kernel);
        }
        if (start_kernel != nullptr) {
            cudaEventDestroy(start_kernel);
        }
        if (d_lut != nullptr) {
            cudaFree(d_lut);
        }
        if (d_histogram != nullptr) {
            cudaFree(d_histogram);
        }
        if (d_edge != nullptr) {
            cudaFree(d_edge);
        }
        if (d_equalized != nullptr) {
            cudaFree(d_equalized);
        }
        if (d_blur != nullptr) {
            cudaFree(d_blur);
        }
        if (d_gray != nullptr) {
            cudaFree(d_gray);
        }
        if (d_rgb != nullptr) {
            cudaFree(d_rgb);
        }
    };

    try {
        print_gpu_device_info();
        std::cout << "Starting G2ES GPU Image Pipeline..." << std::endl;

        const int width = input_image.width;
        const int height = input_image.height;
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const size_t rgb_bytes = gray_bytes * 3;
        PipelineOutputs outputs = make_outputs(pixels);

        if (input_image.channels == 3) {
            MAIN_CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
        }
        MAIN_CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
        MAIN_CUDA_CHECK(cudaMalloc(&d_blur, gray_bytes));
        MAIN_CUDA_CHECK(cudaMalloc(&d_equalized, gray_bytes));
        MAIN_CUDA_CHECK(cudaMalloc(&d_edge, gray_bytes));
        MAIN_CUDA_CHECK(cudaMalloc(&d_histogram, kHistogramBins * sizeof(unsigned int)));
        MAIN_CUDA_CHECK(cudaMalloc(&d_lut, kHistogramBins * sizeof(unsigned char)));

        MAIN_CUDA_CHECK(cudaEventCreate(&start_kernel));
        MAIN_CUDA_CHECK(cudaEventCreate(&stop_kernel));
        MAIN_CUDA_CHECK(cudaEventCreate(&start_e2e));
        MAIN_CUDA_CHECK(cudaEventCreate(&stop_e2e));

        MAIN_CUDA_CHECK(cudaEventRecord(start_e2e));

        if (input_image.channels == 3) {
            MAIN_CUDA_CHECK(cudaMemcpy(d_rgb,
                                       input_image.pixels.data(),
                                       rgb_bytes,
                                       cudaMemcpyHostToDevice));
        } else {
            MAIN_CUDA_CHECK(cudaMemcpy(d_gray,
                                       input_image.pixels.data(),
                                       gray_bytes,
                                       cudaMemcpyHostToDevice));
        }

        const dim3 block(kBlockSize, kBlockSize);
        const dim3 grid(div_up(width, block.x), div_up(height, block.y));
        const int threads = 256;
        const int linear_blocks = div_up(pixels, threads);

        MAIN_CUDA_CHECK(cudaEventRecord(start_kernel));

        if (input_image.channels == 3) {
            rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
            MAIN_CUDA_CHECK(cudaGetLastError());
        }

        gaussian_blur_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
        MAIN_CUDA_CHECK(cudaGetLastError());

        MAIN_CUDA_CHECK(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
        compute_histogram_global_kernel<<<linear_blocks, threads>>>(d_blur, d_histogram, pixels);
        MAIN_CUDA_CHECK(cudaGetLastError());

        build_equalization_lut_kernel<<<1, kHistogramBins>>>(d_histogram, d_lut, pixels);
        MAIN_CUDA_CHECK(cudaGetLastError());

        apply_lut_kernel<<<linear_blocks, threads>>>(d_blur, d_equalized, d_lut, pixels);
        MAIN_CUDA_CHECK(cudaGetLastError());

        sobel_edge_naive_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
        MAIN_CUDA_CHECK(cudaGetLastError());

        MAIN_CUDA_CHECK(cudaEventRecord(stop_kernel));
        MAIN_CUDA_CHECK(cudaEventSynchronize(stop_kernel));

        MAIN_CUDA_CHECK(cudaMemcpy(outputs.gray.data(),
                                   d_gray,
                                   gray_bytes,
                                   cudaMemcpyDeviceToHost));
        MAIN_CUDA_CHECK(cudaMemcpy(outputs.blur.data(),
                                   d_blur,
                                   gray_bytes,
                                   cudaMemcpyDeviceToHost));
        MAIN_CUDA_CHECK(cudaMemcpy(outputs.equalized.data(),
                                   d_equalized,
                                   gray_bytes,
                                   cudaMemcpyDeviceToHost));
        MAIN_CUDA_CHECK(cudaMemcpy(outputs.edge.data(),
                                   d_edge,
                                   gray_bytes,
                                   cudaMemcpyDeviceToHost));

        MAIN_CUDA_CHECK(cudaEventRecord(stop_e2e));
        MAIN_CUDA_CHECK(cudaEventSynchronize(stop_e2e));

        float kernel_time_ms = 0.0f;
        float e2e_time_ms = 0.0f;
        MAIN_CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel));
        MAIN_CUDA_CHECK(cudaEventElapsedTime(&e2e_time_ms, start_e2e, stop_e2e));

        const double write_ms = write_pipeline_outputs(output_prefix, outputs, width, height);

        std::cout << "\n=== GPU Timing Results ===" << std::endl;
        std::cout << "GPU kernel total:      " << kernel_time_ms << " ms" << std::endl;
        std::cout << "GPU transfer+kernel:   " << e2e_time_ms << " ms" << std::endl;
        std::cout << "GPU write outputs:     " << write_ms << " ms" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "GPU outputs written with prefix: " << output_prefix << std::endl;

        cleanup();
        return GpuTiming{kernel_time_ms, e2e_time_ms, write_ms};
    } catch (...) {
        cleanup();
        throw;
    }
}

std::string cpu_output_prefix(const Options& options) {
    if (options.mode == RunMode::kBoth) {
        return options.output_prefix + "_cpu";
    }
    if (options.output_prefix_provided) {
        return options.output_prefix;
    }
    return options.output_prefix + "_cpu";
}

std::string gpu_output_prefix(const Options& options) {
    if (options.mode == RunMode::kBoth) {
        return options.output_prefix + "_gpu";
    }
    return options.output_prefix;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }

        const auto load_start = Clock::now();
        const ImageData input_image = load_image(options.input_path);
        const auto load_end = Clock::now();

        const int pixels = input_image.width * input_image.height;
        std::cout << "Input image: " << options.input_path << std::endl;
        std::cout << "Image size:  " << input_image.width << " x " << input_image.height
                  << " (" << pixels << " pixels)" << std::endl;
        std::cout << "Load image:  " << elapsed_ms(load_start, load_end) << " ms"
                  << std::endl;

        CpuTiming cpu_timing;
        GpuTiming gpu_timing;
        bool ran_cpu = false;
        bool ran_gpu = false;

        if (options.mode == RunMode::kCpu || options.mode == RunMode::kBoth) {
            cpu_timing = run_cpu_pipeline(input_image, cpu_output_prefix(options));
            ran_cpu = true;
        }

        if (options.mode == RunMode::kGpu || options.mode == RunMode::kBoth) {
            gpu_timing = run_gpu_pipeline(input_image, gpu_output_prefix(options));
            ran_gpu = true;
        }

        if (ran_cpu && ran_gpu) {
            std::cout << "\n=== CPU vs GPU Summary ===" << std::endl;
            std::cout << "CPU compute total:     " << cpu_timing.compute_total_ms
                      << " ms" << std::endl;
            std::cout << "GPU kernel total:      " << gpu_timing.kernel_total_ms
                      << " ms" << std::endl;
            std::cout << "GPU transfer+kernel:   "
                      << gpu_timing.transfer_and_kernel_ms << " ms" << std::endl;
            if (gpu_timing.kernel_total_ms > 0.0f) {
                std::cout << "Speedup vs kernels:    "
                          << cpu_timing.compute_total_ms / gpu_timing.kernel_total_ms
                          << "x" << std::endl;
            }
            if (gpu_timing.transfer_and_kernel_ms > 0.0f) {
                std::cout << "Speedup vs transfers:  "
                          << cpu_timing.compute_total_ms /
                                 gpu_timing.transfer_and_kernel_ms
                          << "x" << std::endl;
            }
            std::cout << "==========================" << std::endl;
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        print_usage(argv[0]);
        std::cerr << "Pipeline failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
