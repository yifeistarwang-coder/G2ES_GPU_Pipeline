/**
 * @file sobel_edge.cu
 * @brief 使用CUDA实现Sobel边缘检测
 */

#include "kernels.cuh"
#include "utils.h"

namespace {

constexpr int kSobelRadius = 1;
constexpr int kSobelOptimizedBlockWidth = 16;
constexpr int kSobelOptimizedBlockHeight = 16;
constexpr int kSobelTileWidth = kSobelOptimizedBlockWidth + 2 * kSobelRadius;
constexpr int kSobelTileHeight = kSobelOptimizedBlockHeight + 2 * kSobelRadius;

}  // namespace

/**
 * @brief Sobel边缘检测CUDA核函数（朴素实现）
 * @param input 输入图像数据（灰度图）
 * @param output 输出边缘检测结果
 * @param width 图像宽度
 * @param height 图像高度
 * @details 使用3x3 Sobel算子计算图像的水平和垂直梯度，然后计算梯度幅值
 *          Sobel算子是一种离散微分算子，用于计算图像亮度函数的梯度近似值
 *          边缘像素（最外圈）由于缺乏完整邻域，直接设为0
 */
__global__ void sobel_edge_naive_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width,
                                        int height) {
    // 计算当前线程对应的像素坐标
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 检查是否超出图像边界
    if (x >= width || y >= height) {
        return;
    }

    // 边界像素不做Sobel计算，直接输出0（因为算子需要3x3邻域）
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    // 计算水平方向梯度Gx（Sobel水平算子）
    // 算子矩阵:
    // [ -1  0  1 ]
    // [ -2  0  2 ]
    // [ -1  0  1 ]
    const int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
                   - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
                   - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    // 计算垂直方向梯度Gy（Sobel垂直算子）
    // 算子矩阵:
    // [ -1 -2 -1 ]
    // [  0  0  0 ]
    // [  1  2  1 ]
    const int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x]
                   - input[(y - 1) * width + (x + 1)]
                   + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x]
                   + input[(y + 1) * width + (x + 1)];

    // 计算梯度幅值: magnitude = sqrt(Gx^2 + Gy^2)
    // 将结果截断在0-255范围内，作为边缘强度输出
    output[y * width + x] = static_cast<unsigned char>(
        min(255, static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)))));
}

/**
 * @brief Sobel边缘检测CUDA核函数（共享内存优化版）
 * @param input 输入图像数据（灰度图）
 * @param output 输出边缘检测结果
 * @param width 图像宽度
 * @param height 图像高度
 * @details 每个block协作加载16x16输出区域及其一圈halo到共享内存，减少相邻线程
 *          对global memory的重复读取。边界处理和朴素版本保持一致：最外圈输出0。
 *
 *          实验结论（sm_87 / Jetson AGX Orin / 1440x810图像）:
 *          - 共享内存int tile版本比朴素版慢~8%：协作加载+同步的固定开销 > 3x3小核的带宽节省
 *          - unsigned char tile比int tile更慢（~2x）：行宽18字节导致共享内存bank conflict加剧
 *          - __ldg()比普通load更慢：sm_87上绕过L1走texture cache路径，延迟更高
 *          - uchar4向量化：协作加载步长=block_threads时其他线程会覆盖uchar4写的后3个字节
 *          结论：3x3 Sobel核太小，任何共享内存优化在此硬件上都没有收益，朴素版即最优。
 */
__global__ void sobel_edge_optimized_kernel(const unsigned char* input,
                                            unsigned char* output,
                                            int width,
                                            int height) {
    // 静态共享内存按当前管线使用的16x16 block设计。
    if (blockDim.x > kSobelOptimizedBlockWidth ||
        blockDim.y > kSobelOptimizedBlockHeight) {
        return;
    }

    __shared__ int tile[kSobelTileHeight][kSobelTileWidth];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;

    const int tile_width = blockDim.x + 2 * kSobelRadius;
    const int tile_height = blockDim.y + 2 * kSobelRadius;
    const int tile_origin_x = blockIdx.x * blockDim.x - kSobelRadius;
    const int tile_origin_y = blockIdx.y * blockDim.y - kSobelRadius;
    const int thread_linear = ty * blockDim.x + tx;
    const int block_threads = blockDim.x * blockDim.y;

    // 协作加载当前block需要的输入像素和halo。图像外区域填0，避免越界读取。
    for (int index = thread_linear; index < tile_width * tile_height;
         index += block_threads) {
        const int tile_y = index / tile_width;
        const int tile_x = index % tile_width;
        const int image_x = tile_origin_x + tile_x;
        const int image_y = tile_origin_y + tile_y;

        tile[tile_y][tile_x] =
            (image_x >= 0 && image_x < width && image_y >= 0 && image_y < height)
                ? static_cast<int>(input[image_y * width + image_x])
                : 0;
    }
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    const int center_x = tx + kSobelRadius;
    const int center_y = ty + kSobelRadius;

    const int top_left = tile[center_y - 1][center_x - 1];
    const int top = tile[center_y - 1][center_x];
    const int top_right = tile[center_y - 1][center_x + 1];
    const int left = tile[center_y][center_x - 1];
    const int right = tile[center_y][center_x + 1];
    const int bottom_left = tile[center_y + 1][center_x - 1];
    const int bottom = tile[center_y + 1][center_x];
    const int bottom_right = tile[center_y + 1][center_x + 1];

    const int gx = -top_left + top_right - 2 * left + 2 * right -
                   bottom_left + bottom_right;
    const int gy = -top_left - 2 * top - top_right + bottom_left +
                   2 * bottom + bottom_right;
    const int magnitude = static_cast<int>(
        sqrtf(static_cast<float>(gx * gx + gy * gy)));

    output[y * width + x] = static_cast<unsigned char>(min(255, magnitude));
}


