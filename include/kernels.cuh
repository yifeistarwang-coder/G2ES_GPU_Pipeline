#ifndef KERNELS_CUH
#define KERNELS_CUH

/**
 * @brief RGB转灰度图的CUDA核函数
 * @param rgb 输入的RGB图像数据（每个像素3字节）
 * @param gray 输出的灰度图数据（每个像素1字节）
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void rgb_to_gray_kernel(unsigned char* rgb, unsigned char* gray, int width, int height);
/*
优化
*/
__global__ void rgb_to_gray_optimized_kernel(const uchar3* rgb, unsigned char* gray, int width, int height);

/**
 * @brief 高斯模糊的CUDA核函数
 * @param input 输入图像数据
 * @param output 输出图像数据
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height);

/*
优化：共享内存 + 可分离滤波
*/
__global__ void gaussian_blur_optimized_kernel(unsigned char* input, unsigned char* output, int width, int height);

/**
 * @brief 计算图像直方图的CUDA核函数（使用全局内存）
 * @param image 输入图像数据
 * @param histogram 输出的直方图数组（256个bin）
 * @param pixels 图像总像素数
 */
__global__ void compute_histogram_global_kernel(const unsigned char* image,
                                                unsigned int* histogram,
                                                int pixels);

/*
优化：共享内存局部直方图 + grid-stride loop
*/
__global__ void compute_histogram_optimized_kernel(const unsigned char* image,
                                                   unsigned int* histogram,
                                                   int pixels);

/**
 * @brief 构建直方图均衡化查找表(LUT)的CUDA核函数
 * @param histogram 输入的直方图数据
 * @param lut 输出的查找表
 * @param pixels 图像总像素数
 */
__global__ void build_equalization_lut_kernel(const unsigned int* histogram,
                                              unsigned char* lut,
                                              int pixels);

/**
 * @brief 应用查找表进行像素映射的CUDA核函数
 * @param input 输入图像数据
 * @param output 输出图像数据
 * @param lut 查找表
 * @param pixels 图像总像素数
 */
__global__ void apply_lut_kernel(const unsigned char* input,
                                 unsigned char* output,
                                 const unsigned char* lut,
                                 int pixels);

/**
 * @brief Sobel边缘检测的CUDA核函数（朴素实现）
 * @param input 输入图像数据
 * @param output 输出边缘检测结果
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void sobel_edge_naive_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width,
                                        int height);

/**
 * @brief Sobel边缘检测的CUDA核函数（共享内存优化版）
 * @param input 输入图像数据
 * @param output 输出边缘检测结果
 * @param width 图像宽度
 * @param height 图像高度
 */
__global__ void sobel_edge_optimized_kernel(const unsigned char* input,
                                            unsigned char* output,
                                            int width,
                                            int height);

#endif
