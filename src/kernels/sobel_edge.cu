#include "kernels.h"
#include "utils.h"

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

    // 边界像素不做 Sobel 计算，直接输出 0（因为算子需要 3x3 邻域）
    if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        output[y * width + x] = 0;
        return;
    }

    // 计算水平方向梯度 Gx
    // [ -1  0  1 ]
    // [ -2  0  2 ]
    // [ -1  0  1 ]
    const int gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
                   - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
                   - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];
    
    // 计算垂直方向梯度 Gy
    // [ -1 -2 -1 ]
    // [  0  0  0 ]
    // [  1  2  1 ]
    const int gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x]
                   - input[(y - 1) * width + (x + 1)]
                   + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x]
                   + input[(y + 1) * width + (x + 1)];

    // 计算梯度幅值：sqrt(Gx^2 + Gy^2)，并将结果截断在 0-255 范围内
    output[y * width + x] = static_cast<unsigned char>(
        min(255, static_cast<int>(sqrtf(static_cast<float>(gx * gx + gy * gy)))));
}