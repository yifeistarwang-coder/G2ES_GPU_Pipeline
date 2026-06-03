#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief CUDA错误检查宏
 * @param val CUDA API调用的返回值
 * @details 自动检查CUDA调用是否成功，失败时输出错误信息并终止程序
 */
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

/**
 * @brief CUDA错误检查函数
 * @param result CUDA API返回的错误码
 * @param func 调用CUDA函数的名称
 * @param file 调用所在的源文件路径
 * @param line 调用所在的行号
 */
void check(cudaError_t result, char const *const func, const char *const file, int const line);

#endif
