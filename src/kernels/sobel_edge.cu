/**
 * @file sobel_edge.cu
 * @brief 使用CUDA实现Sobel边缘检测
 */

#include "kernels.h"
#include "utils.h"

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