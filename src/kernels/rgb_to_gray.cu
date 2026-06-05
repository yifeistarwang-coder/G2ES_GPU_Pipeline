/**
 * @file rgb_to_gray.cu
 * @brief 使用CUDA实现RGB转灰度图
 */

#include "kernels.cuh"
#include "utils.h"

/**
 * @brief RGB转灰度图CUDA核函数
 * @param rgb 输入的RGB图像数据（每个像素3字节：R、G、B）
 * @param gray 输出的灰度图数据（每个像素1字节）
 * @param width 图像宽度
 * @param height 图像高度
 * @details 使用加权公式: Gray = 0.299 * R + 0.587 * G + 0.114 * B
 *          这是ITU-R BT.601标准的亮度计算公式，符合人眼对不同颜色的敏感度
 */
__global__ void rgb_to_gray_kernel(unsigned char* rgb,
                                    unsigned char* gray,
                                    int width,
                                    int height)
{
    // 将当前线程在线程块中的局部位置换算为图像中的全局二维坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：确保线程在图像范围内
    if(x >= width || y >= height) {
        return;
    }

    // 计算当前像素在一维数组中的索引位置
    const int pixel_index = y * width + x;  // 灰度图像素索引（1字节/像素）
    const int idx = pixel_index * 3;        // RGB图像素索引（3字节/像素）

    // 将无符号字符转换为浮点数，以便进行精确的灰度值计算
    const float r = static_cast<float>(rgb[idx + 0]);  // 红色通道
    const float g = static_cast<float>(rgb[idx + 1]);  // 绿色通道
    const float b = static_cast<float>(rgb[idx + 2]);  // 蓝色通道

    // 使用ITU-R BT.601标准公式计算灰度值
    // 人眼对绿色最敏感，红色次之，蓝色最不敏感
    const float gray_value = 0.299f * r + 0.587f * g + 0.114f * b;

    // 将计算结果存储到灰度图数组中
    gray[pixel_index] = static_cast<unsigned char>(gray_value);
}

__global__ void rgb_to_gray_optimized_kernel(const uchar3* rgb,
                                    unsigned char* gray,
                                    int width,
                                    int height){
    
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }
    const int pixel_index = y * width + x;  // 灰度图像素索引（1字节/像素）
    const uchar3 pixel = rgb[pixel_index];  // 直接读取RGB像素数据

    gray[pixel_index] = static_cast<unsigned char>(77 * pixel.x + 150 * pixel.y + 29 * pixel.z >> 8);
}