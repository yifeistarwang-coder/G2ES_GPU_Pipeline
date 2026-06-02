#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t result, char const *const func, const char *const file, int const line);

#endif
