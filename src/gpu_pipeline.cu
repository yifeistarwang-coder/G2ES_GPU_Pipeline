#include "gpu_pipeline.h"

#include <cuda_runtime.h>

#include "kernels.h"
#include "utils.h"

void print_gpu_device_info() {
    int device_id = 0;
    CHECK_CUDA_ERROR(cudaGetDevice(&device_id));

    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device_id));

    int clock_khz = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id));

    int mem_clock_khz = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&mem_clock_khz,
                                           cudaDevAttrMemoryClockRate,
                                           device_id));

    int mem_bus_width = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&mem_bus_width,
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
        if (stop_e2e != nullptr) cudaEventDestroy(stop_e2e);
        if (start_e2e != nullptr) cudaEventDestroy(start_e2e);
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
        print_gpu_device_info();
        std::cout << "Starting G2ES GPU Image Pipeline..." << std::endl;

        const int width = input_image.width;
        const int height = input_image.height;
        const int pixels = width * height;
        const size_t gray_bytes = static_cast<size_t>(pixels) * sizeof(unsigned char);
        const size_t rgb_bytes = gray_bytes * 3;
        PipelineOutputs outputs = make_outputs(pixels);

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
        CHECK_CUDA_ERROR(cudaEventCreate(&start_e2e));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop_e2e));

        CHECK_CUDA_ERROR(cudaEventRecord(start_e2e));

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

        if (input_image.channels == 3) {
            rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        gaussian_blur_kernel<<<grid, block>>>(d_gray, d_blur, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, kHistogramBins * sizeof(unsigned int)));
        compute_histogram_global_kernel<<<linear_blocks, threads>>>(d_blur, d_histogram, pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());

        build_equalization_lut_kernel<<<1, kHistogramBins>>>(d_histogram, d_lut, pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());

        apply_lut_kernel<<<linear_blocks, threads>>>(d_blur, d_equalized, d_lut, pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());

        sobel_edge_naive_kernel<<<grid, block>>>(d_equalized, d_edge, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());

        CHECK_CUDA_ERROR(cudaEventRecord(stop_kernel));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_kernel));

        CHECK_CUDA_ERROR(cudaMemcpy(outputs.gray.data(), d_gray, gray_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(outputs.blur.data(), d_blur, gray_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(outputs.equalized.data(), d_equalized, gray_bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR(cudaMemcpy(outputs.edge.data(), d_edge, gray_bytes, cudaMemcpyDeviceToHost));

        CHECK_CUDA_ERROR(cudaEventRecord(stop_e2e));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop_e2e));

        float kernel_time_ms = 0.0f;
        float e2e_time_ms = 0.0f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_time_ms, start_kernel, stop_kernel));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&e2e_time_ms, start_e2e, stop_e2e));

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
