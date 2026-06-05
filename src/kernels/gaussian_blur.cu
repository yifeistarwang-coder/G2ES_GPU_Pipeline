/**
 * @file gaussian_blur.cu
 * @brief 使用CUDA实现高斯模糊滤波器
 */

#include "kernels.cuh"
#include "utils.h"

namespace {

// 5x5高斯核定义
constexpr int kGaussianRadius = 2;  // 高斯核半径
constexpr int kGaussianWidth = 2 * kGaussianRadius + 1;  // 高斯核宽度 = 5
constexpr int kGaussianWeightSum = 159;  // 高斯核所有元素之和，用于归一化
constexpr int kGaussian1DWeightSum = 17;  // 1D可分离核权重和
constexpr int kOptimizedBlockWidth = 16;
constexpr int kOptimizedBlockHeight = 16;
constexpr int kOptimizedTileWidth = kOptimizedBlockWidth + 2 * kGaussianRadius;
constexpr int kOptimizedTileHeight = kOptimizedBlockHeight + 2 * kGaussianRadius;

// 存储在常量内存中的5x5高斯核权重
__constant__ int kGaussian5x5[kGaussianWidth * kGaussianWidth] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

// 可分离高斯核，横向和纵向各做一次
__constant__ int kGaussian1D[kGaussianWidth] = {2, 4, 5, 4, 2};

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

//共享内存 + 可分离滤波
__global__ void gaussian_blur_optimized_kernel(unsigned char* input,
                                               unsigned char* output,
                                               int width,
                                               int height) {
    // 静态共享内存按当前管线使用的16x16 block设计。
    if (blockDim.x > kOptimizedBlockWidth || blockDim.y > kOptimizedBlockHeight) {
        return;
    }

    __shared__ unsigned char tile[kOptimizedTileHeight][kOptimizedTileWidth];
    __shared__ int horizontal[kOptimizedTileHeight][kOptimizedBlockWidth];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    const int tile_width = blockDim.x + 2 * kGaussianRadius;
    const int tile_height = blockDim.y + 2 * kGaussianRadius;
    const int tile_origin_x = blockIdx.x * blockDim.x - kGaussianRadius;
    const int tile_origin_y = blockIdx.y * blockDim.y - kGaussianRadius;
    const int thread_linear = ty * blockDim.x + tx;
    const int block_threads = blockDim.x * blockDim.y;

    // 协作加载当前block需要的输入像素和halo，边界使用clamp方式处理。
    for (int index = thread_linear; index < tile_width * tile_height;
         index += block_threads) {
        const int tile_y = index / tile_width;
        const int tile_x = index % tile_width;
        const int image_x = clamp_device(tile_origin_x + tile_x, 0, width - 1);
        const int image_y = clamp_device(tile_origin_y + tile_y, 0, height - 1);

        tile[tile_y][tile_x] = input[image_y * width + image_x];
    }
    __syncthreads();

    // 第一阶段：对共享内存中的每一行做横向1D高斯滤波。
    for (int tile_y = ty; tile_y < tile_height; tile_y += blockDim.y) {
        int sum = 0;
        const int center_x = tx + kGaussianRadius;

        for (int kx = -kGaussianRadius; kx <= kGaussianRadius; ++kx) {
            const int kernel_index = kx + kGaussianRadius;
            sum += static_cast<int>(tile[tile_y][center_x + kx]) *
                   kGaussian1D[kernel_index];
        }

        horizontal[tile_y][tx] = sum;
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    // 第二阶段：对横向滤波结果做纵向1D高斯滤波，并一次性归一化。
    int sum = 0;
    const int center_y = ty + kGaussianRadius;
    for (int ky = -kGaussianRadius; ky <= kGaussianRadius; ++ky) {
        const int kernel_index = ky + kGaussianRadius;
        sum += horizontal[center_y + ky][tx] * kGaussian1D[kernel_index];
    }

    constexpr int weight_sum = kGaussian1DWeightSum * kGaussian1DWeightSum;
    output[y * width + x] = static_cast<unsigned char>((sum + weight_sum / 2) / weight_sum);
}
