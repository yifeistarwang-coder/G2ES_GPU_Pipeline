#include "kernels.h"
#include "utils.h"

// Gray = 0.299 * R + 0.587 * G + 0.114 * B


__global__ void rgb_to_gray_kernel(unsigned char* rgb, 
                                    unsigned char* gray,
                                    int width, 
                                    int height)
{
    // 将当前线程在线程块中的局部位置换算为图像中的全局二维坐标。
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

    // 计算当前像素在一维数组中的索引位置。
    const int pixel_index = y * width + x; // 灰度图
    const int idx = pixel_index * 3; // RGB图 （3个字节 R，G，B）

    //static_cast<float> 将无符号字符转换为浮点数，以便进行灰度值的计算。
    const float r = static_cast<float>(rgb[idx + 0]);
    const float g = static_cast<float>(rgb[idx + 1]);
    const float b = static_cast<float>(rgb[idx + 2]);

    // 计算灰度值
    const float gray_value = 0.299f * r + 0.587f * g + 0.114f * b;

    // 将结果存储到灰度图数组中
    gray[pixel_index] = static_cast<unsigned char>(gray_value);

} 
