#ifndef KERNELS_H
#define KERNELS_H

constexpr int kHistogramBins = 256;

__global__ void rgb_to_gray_kernel(unsigned char* rgb, unsigned char* gray, int width, int height);
__global__ void gaussian_blur_kernel(unsigned char* input, unsigned char* output, int width, int height);
__global__ void compute_histogram_global_kernel(const unsigned char* image,
                                                unsigned int* histogram,
                                                int pixels);
__global__ void build_equalization_lut_kernel(const unsigned int* histogram,
                                              unsigned char* lut,
                                              int pixels);
__global__ void apply_lut_kernel(const unsigned char* input,
                                 unsigned char* output,
                                 const unsigned char* lut,
                                 int pixels);
__global__ void sobel_edge_naive_kernel(const unsigned char* input,
                                        unsigned char* output,
                                        int width,
                                        int height);

#endif
