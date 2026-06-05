#ifndef PIPELINE_COMMON_H
#define PIPELINE_COMMON_H

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// 直方图的bin数量，对应0-255共256个灰度级
constexpr int kHistogramBins = 256;
// CUDA线程块大小 (16x16 = 256个线程)
constexpr int kBlockSize = 16;
// 高斯核半径
constexpr int kGaussianRadius = 2;
// 高斯核宽度 = 2*半径+1 = 5
constexpr int kGaussianWidth = 2 * kGaussianRadius + 1;
// 高斯核权重总和，用于归一化
constexpr int kGaussianWeightSum = 159;
// 默认输入图像路径
constexpr const char* kDefaultInputPath = "image/test_image.png";

// 5x5高斯模糊核（近似高斯分布）
const int kGaussian5x5[kGaussianWidth * kGaussianWidth] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

// 运行模式枚举
enum class RunMode {
    kGpu,           // 仅GPU模式（naive kernels）
    kCpu,           // 仅CPU模式
    kBoth,          // CPU和GPU对比模式
    kGpuOptimized,  // GPU模式（optimized kernels）
    kGpuBenchmark   // GPU性能基准测试模式
};

// 命令行选项结构体
struct Options {
    RunMode mode = RunMode::kGpu;           // 运行模式
    std::string input_path = kDefaultInputPath; // 输入图像路径
    std::string output_prefix;              // 输出文件前缀
    bool output_prefix_provided = false;    // 是否提供了输出前缀
    bool show_help = false;                 // 是否显示帮助信息
};

// 图像数据结构体
struct ImageData {
    int width = 0;                          // 图像宽度
    int height = 0;                         // 图像高度
    int channels = 0;                       // 通道数（1=灰度，3=RGB）
    std::vector<unsigned char> pixels;      // 像素数据
};

// 管线各阶段输出数据
struct PipelineOutputs {
    std::vector<unsigned char> gray;        // 灰度图输出
    std::vector<unsigned char> blur;        // 高斯模糊输出
    std::vector<unsigned char> equalized;   // 直方图均衡化输出
    std::vector<unsigned char> edge;        // 边缘检测输出
};

// CPU计时结果
struct CpuTiming {
    double compute_total_ms = 0.0;          // 计算总耗时（毫秒）
    double write_ms = 0.0;                  // 文件写入耗时（毫秒）
};

// GPU计时结果
struct GpuTiming {
    float kernel_total_ms = 0.0f;           // 核函数总耗时（毫秒）
    float transfer_and_kernel_ms = 0.0f;    // 数据传输+核函数耗时（毫秒）
    double write_ms = 0.0;                  // 文件写入耗时（毫秒）
};

// 时钟类型别名
using Clock = std::chrono::steady_clock;

// ==================== 内联工具函数 ====================

/**
 * @brief 计算两个时间点之间的毫秒差
 */
inline double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief 向上整除函数
 */
inline int div_up(const int value, const int divisor) {
    return (value + divisor - 1) / divisor;
}

/**
 * @brief 整数钳位函数
 */
inline int clamp_int(const int value, const int low, const int high) {
    return std::max(low, std::min(value, high));
}

/**
 * @brief 去除文件路径的扩展名
 */
inline std::string strip_extension(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        return path;
    }
    return path.substr(0, dot);
}

// ==================== 函数声明 ====================

void print_usage(const char* program_name);
bool parse_mode_flag(const std::string& value, RunMode* mode);
Options parse_options(int argc, char** argv);
ImageData load_image(const std::string& path);
void write_png(const std::string& path,
               const std::vector<unsigned char>& pixels,
               int width,
               int height);
double write_pipeline_outputs(const std::string& output_prefix,
                              const PipelineOutputs& outputs,
                              int width,
                              int height);
PipelineOutputs make_outputs(int pixels);
void print_cpu_info();

#endif
