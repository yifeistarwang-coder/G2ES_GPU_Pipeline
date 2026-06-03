#ifndef GPU_PIPELINE_H
#define GPU_PIPELINE_H

#include "pipeline_common.h"

/**
 * @brief 打印GPU设备信息
 */
void print_gpu_device_info();

/**
 * @brief 运行GPU图像处理管线
 * @param input_image 输入图像数据
 * @param output_prefix 输出文件前缀
 * @return GPU计时结果
 */
GpuTiming run_gpu_pipeline(const ImageData& input_image,
                           const std::string& output_prefix);

#endif
