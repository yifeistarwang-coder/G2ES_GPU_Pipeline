/**
 * @file gaussian_blur.cu
 * @brief 使用CUDA实现高斯模糊滤波器
 */

#include "kernels.h"
#include "utils.h"

namespace {

// 5x5高斯核定义
constexpr int kGaussianRadius = 2;  // 高斯核半径
constexpr int kGaussianWidth = 2 * kGaussianRadius + 1;  // 高斯核宽度 = 5
constexpr int kGaussianWeightSum = 159;  // 高斯核所有元素之和，用于归一化

// 存储在常量内存中的5x5高斯核权重
__constant__ int kGaussian5x5[kGaussianWidth * kGaussianWidth] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

/**
 * @brief 设备端钳位函数，将值限制在[low, high]范围内
 * @param value 输入值
 * @param low 下界
 * @param high 上界
 * @return 钳位后的值
 */
__device__ __forceinline__ int clamp_device(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

}  // namespace

/**
 * @brief 高斯模糊CUDA核函数
 * @param input 输入图像数据（灰度图）
 * @param output 输出图像数据
 * @param width 图像宽度
 * @param height 图像高度
 * @details 每个线程处理一个像素，使用5x5高斯核进行卷积，边界使用clamp方式处理
 */
__global__ void gaussian_blur_kernel(unsigned char* input,
                                     unsigned char* output,
                                     int width,
                                     int height) {

    // 计算当前线程处理的像素坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查线程是否在图像边界内
    if (x >= width || y >= height) {
        return;
    }

    int sum = 0;
    // 遍历5x5高斯核区域
    for (int ky = -kGaussianRadius; ky <= kGaussianRadius; ++ky) {
        // 将Y坐标钳位到有效范围，处理边界情况
        const int yy = clamp_device(y + ky, 0, height - 1);
        for (int kx = -kGaussianRadius; kx <= kGaussianRadius; ++kx) {
            // 将X坐标钳位到有效范围，处理边界情况
            const int xx = clamp_device(x + kx, 0, width - 1);

            // 计算高斯核权重索引
            const int kernel_index = (ky + kGaussianRadius) * kGaussianWidth +
                                     (kx + kGaussianRadius);

            // 累加加权像素值
            sum += static_cast<int>(input[yy * width + xx]) * kGaussian5x5[kernel_index];
        }
    }

    // 归一化并存储结果（带四舍五入）
    output[y * width + x] = static_cast<unsigned char>((sum + kGaussianWeightSum / 2) /
                                                       kGaussianWeightSum);
}
