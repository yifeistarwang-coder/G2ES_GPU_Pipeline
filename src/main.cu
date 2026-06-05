/**
 * G2ES 图像处理管线
 *
 * 处理流程: RGB -> 灰度图 -> 高斯模糊 -> 直方图均衡化 -> Sobel边缘检测
 *
 * 使用方法:
 *   ./image_pipeline [--gpu|--cpu|--both] [输入图像] [输出前缀]
 */

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "pipeline_common.h"
#include "cpu_pipeline.h"
#include "gpu_pipeline.h"

namespace {

/**
 * @brief 生成CPU输出文件的前缀
 */
std::string cpu_output_prefix(const Options& options) {
    if (options.mode == RunMode::kBoth) {
        return options.output_prefix + "_cpu";
    }
    if (options.output_prefix_provided) {
        return options.output_prefix;
    }
    return options.output_prefix + "_cpu";
}

/**
 * @brief 生成GPU输出文件的前缀
 */
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

        // 确保输出目录存在
        const std::filesystem::path prefix_path(options.output_prefix);
        if (prefix_path.has_parent_path()) {
            std::filesystem::create_directories(prefix_path.parent_path());
        }

        print_cpu_info();

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

        if (options.mode == RunMode::kGpuOptimized) {
            gpu_timing = run_gpu_optimized_pipeline(input_image, gpu_output_prefix(options));
            ran_gpu = true;
        }

        if (options.mode == RunMode::kGpuBenchmark) {
            run_gpu_benchmark(input_image);
            return EXIT_SUCCESS;
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
