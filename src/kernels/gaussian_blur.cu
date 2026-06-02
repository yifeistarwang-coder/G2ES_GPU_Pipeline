/**
 * @file gaussian_blur.cu
 * @brief Implement Gaussian Blur filter using CUDA.
 */

#include "kernels.h"
#include "utils.h"

namespace {

// Definition for 5x5 Gaussian kernel
constexpr int kGaussianRadius = 2;
constexpr int kGaussianWidth = 2 * kGaussianRadius + 1;
constexpr int kGaussianWeightSum = 159; // Sum of all elements in the 5x5 kernel

// Normalization factor is 159 for this 5x5 integer Gaussian kernel
__constant__ int kGaussian5x5[kGaussianWidth * kGaussianWidth] = {
    2, 4, 5, 4, 2,
    4, 9, 12, 9, 4,
    5, 12, 15, 12, 5,
    4, 9, 12, 9, 4,
    2, 4, 5, 4, 2
};

/**
 * @brief Utility function to clamp a value between low and high.
 */
__device__ __forceinline__ int clamp_device(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

}  // namespace

/**
 * @brief Gaussian blur kernel for image processing.
 * 
 * @param input   Pointer to the input image data (grayscale).
 * @param output  Pointer to the output image data.
 * @param width   Width of the image.
 * @param height  Height of the image.
 */
__global__ void gaussian_blur_kernel(unsigned char* input,
                                     unsigned char* output,
                                     int width,
                                     int height) {

    // Calculate global pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within image boundaries
    if (x >= width || y >= height) {
        return;
    }

    int sum = 0;
    // Iterate through the 5x5 kernel
    for (int ky = -kGaussianRadius; ky <= kGaussianRadius; ++ky) {    //覆盖5x5区域（外循环）
        // Clamp Y coordinate to handle edge cases
        const int yy = clamp_device(y + ky, 0, height - 1);    //边界处理  
        for (int kx = -kGaussianRadius; kx <= kGaussianRadius; ++kx) {  //覆盖5x5区域（内循环）
            // Clamp X coordinate to handle edge cases
            const int xx = clamp_device(x + kx, 0, width - 1); //边界处理
            
            // Calculate kernel weight index
            const int kernel_index = (ky + kGaussianRadius) * kGaussianWidth +
                                     (kx + kGaussianRadius);
            
            // Add weighted pixel intensity to sum
            sum += static_cast<int>(input[yy * width + xx]) * kGaussian5x5[kernel_index];
        }
    }

    // Divide by weight sum to normalize and store result (with rounding)
    output[y * width + x] = static_cast<unsigned char>((sum + kGaussianWeightSum / 2) /
                                                       kGaussianWeightSum);
}
