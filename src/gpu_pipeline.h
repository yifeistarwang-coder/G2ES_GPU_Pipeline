#ifndef GPU_PIPELINE_H
#define GPU_PIPELINE_H

#include "pipeline_common.h"

/**
 * @brief 打印GPU设备信息
 */
void print_gpu_device_info();

/**
 * @brief 运行GPU图像处理管线（naive kernels）
 * @param input_image 输入图像数据
 * @param output_prefix 输出文件前缀
 * @return GPU计时结果
 */
GpuTiming run_gpu_pipeline(const ImageData& input_image,
                           const std::string& output_prefix);

/**
 * @brief 运行GPU图像处理管线（optimized kernels）
 * @param input_image 输入图像数据
 * @param output_prefix 输出文件前缀
 * @return GPU计时结果
 */
GpuTiming run_gpu_optimized_pipeline(const ImageData& input_image,
                                     const std::string& output_prefix);

/**
 * @brief 运行GPU性能基准测试
 * @param input_image 输入图像数据
 * @details 单独测试每个optimized kernel的性能，使用控制变量法
 *          测试5轮：每轮只用一个optimized kernel，最后测试全部optimized
 */
void run_gpu_benchmark(const ImageData& input_image);

#endif
