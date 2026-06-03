#ifndef CPU_PIPELINE_H
#define CPU_PIPELINE_H

#include "pipeline_common.h"

/**
 * @brief 运行CPU图像处理管线
 * @param input_image 输入图像数据
 * @param output_prefix 输出文件前缀
 * @return CPU计时结果
 */
CpuTiming run_cpu_pipeline(const ImageData& input_image,
                           const std::string& output_prefix);

#endif
