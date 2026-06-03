#include "cpu_pipeline.h"

#include <cmath>

namespace {

/**
 * @brief CPU实现的RGB转灰度图
 * @details 使用加权公式: gray = 0.299*R + 0.587*G + 0.114*B
 */
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

/**
 * @brief CPU实现的高斯模糊
 * @details 使用5x5高斯核进行卷积，边界使用clamp方式处理
 */
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

/**
 * @brief CPU实现的直方图均衡化
 * @details 步骤: 1.计算直方图 2.计算累积分布函数(CDF) 3.构建查找表(LUT) 4.应用LUT
 */
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

/**
 * @brief CPU实现的Sobel边缘检测
 * @details 使用3x3 Sobel算子计算水平和垂直方向的梯度，然后计算梯度幅值
 */
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

}  // namespace

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
