#include "kernels.h"
#include "utils.h"
#include "pipeline_common.h"

/**
 * 计算全局直方图核函数
 *
 * 功能：统计输入图像中每个灰度级出现的次数
 * 原理：每个线程处理一个像素，使用原子操作将对应灰度级的计数器加1
 *
 * @param image    输入图像数据（灰度值范围 0-255）
 * @param histogram 输出直方图数组（256个元素，每个元素对应一个灰度级的计数）
 * @param pixels   图像总像素数
 *
 * 注意：使用 atomicAdd 保证多线程并发写入同一地址时的数据一致性
 */
__global__ void compute_histogram_global_kernel(const unsigned char* image,
                                                unsigned int* histogram,
                                                int pixels) {
    // 计算当前线程处理的像素索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保索引不超过图像像素总数
    if (index < pixels) {
        // 原子操作：将该像素灰度值对应的直方图 bin 计数加 1
        atomicAdd(&histogram[image[index]], 1u);
    }
}

/**
 * 构建直方图均衡化查找表（LUT）核函数
 *
 * 功能：根据直方图计算累积分布函数（CDF），并生成灰度映射表
 * 算法：直方图均衡化 - 增强图像对比度的标准方法
 *
 * 算法步骤：
 * 1. 使用共享内存进行并行前缀和（累加直方图得到 CDF）
 * 2. 找到 CDF 的最小非零值 cdf_min
 * 3. 按照均衡化公式计算每个灰度级的映射值
 *
 * 公式：lut[i] = (cdf[i] - cdf_min) * 255 / (pixels - cdf_min)
 *
 * @param histogram 输入直方图（256个元素）
 * @param lut      输出查找表（256个元素，存储映射后的新灰度值）
 * @param pixels   图像总像素数
 *
 * 优化：使用共享内存避免全局内存访问延迟
 *       使用并行前缀和算法实现高效的累积计算
 */
__global__ void build_equalization_lut_kernel(const unsigned int* histogram,
                                              unsigned char* lut,
                                              int pixels) {
    // 共享内存：存储累积分布函数（CDF），大小为256个灰度级
    __shared__ unsigned int cdf[kHistogramBins];
    // 共享内存：存储CDF的最小非零值
    __shared__ unsigned int cdf_min;

    // 线程索引（0-255），每个线程处理一个灰度级
    const int index = threadIdx.x;

    // 步骤1：将直方图数据加载到共享内存
    cdf[index] = histogram[index];
    // 同步屏障：确保所有线程都完成了数据加载
    __syncthreads();

    // 步骤2：使用并行前缀和（Blelloch扫描算法）计算累积分布函数
    // 原理：通过 log2(n) 步迭代，每步将距离为 stride 的元素相加
    // 最终 cdf[i] 存储的是 histogram[0] + histogram[1] + ... + histogram[i]
    for (int stride = 1; stride < kHistogramBins; stride <<= 1) {
        unsigned int addend = 0;
        // 只有当线程索引大于等于 stride 时，才需要加上前面的值
        if (index >= stride) {
            addend = cdf[index - stride];
        }
        // 同步屏障：确保所有线程读取到正确的旧值
        __syncthreads();
        // 累加操作
        cdf[index] += addend;
        // 同步屏障：确保所有线程完成更新后再进行下一轮
        __syncthreads();
    }

    // 步骤3：线程0负责找到 CDF 的最小非零值
    // 这是直方图均衡化公式中的关键参数，用于避免映射到纯黑
    if (index == 0) {
        cdf_min = 0;
        for (int i = 0; i < kHistogramBins; ++i) {
            if (cdf[i] != 0) {
                cdf_min = cdf[i];
                break;
            }
        }
    }
    // 同步屏障：确保 cdf_min 被正确计算后再被其他线程使用
    __syncthreads();

    // 步骤4：处理特殊情况 - 所有像素都集中在同一个灰度级
    // 此时无法进行均衡化，返回恒等映射（输入即输出）
    if (pixels <= static_cast<int>(cdf_min)) {
        lut[index] = static_cast<unsigned char>(index);
        return;
    }

    // 步骤5：应用直方图均衡化公式
    // 公式推导：将原始 CDF 映射到 [0, 255] 范围
    // numerator = (cdf[i] - cdf_min) * 255
    // denominator = 总像素数 - cdf_min
    const int numerator = static_cast<int>(cdf[index] - cdf_min) * 255;
    const int denominator = pixels - static_cast<int>(cdf_min);
    const int mapped = numerator / denominator;

    // 步骤6：裁剪结果到有效范围 [0, 255] 并存储到查找表
    lut[index] = static_cast<unsigned char>(mapped < 0 ? 0 : (mapped > 255 ? 255 : mapped));
}

/**
 * 应用查找表（LUT）核函数
 *
 * 功能：使用预计算的均衡化 LUT 对每个像素进行映射
 * 优势：避免重复计算，提高处理效率
 *
 * @param input  输入图像数据
 * @param output 输出图像数据（均衡化后的结果）
 * @param lut    预计算的查找表（256个元素）
 * @param pixels 图像总像素数
 *
 * 性能：每个线程处理一个像素，O(1) 时间复杂度
 */
__global__ void apply_lut_kernel(const unsigned char* input,
                                 unsigned char* output,
                                 const unsigned char* lut,
                                 int pixels) {
    // 计算当前线程处理的像素索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查：确保索引不超过图像像素总数
    if (index >= pixels) {
        return;
    }

    // 核心操作：使用查找表进行灰度映射
    // 输入像素值作为索引，查表得到新的灰度值
    // 时间复杂度：O(1)，避免了复杂的数学运算
    output[index] = lut[input[index]];
}
