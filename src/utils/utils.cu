/**
 * @file utils.cu
 * @brief CUDA工具函数实现
 */

#include "utils.h"

/**
 * @brief CUDA错误检查函数
 * @param result CUDA API返回的错误码
 * @param func 调用CUDA函数的名称
 * @param file 调用所在的源文件路径
 * @param line 调用所在的行号
 * @details 检查CUDA调用是否成功，如果失败则输出错误信息并终止程序
 *          错误信息包括：错误码、错误名称、调用位置（文件名和行号）
 */
void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}
