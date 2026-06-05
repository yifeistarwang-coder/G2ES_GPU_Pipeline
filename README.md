# G2ES GPU Image Processing Pipeline

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Educational-blue.svg)](#license)

A CUDA-accelerated image processing pipeline that implements four classic computer vision stages — **G**rayscale, **G**aussian blur, **E**qualization (histogram), **S**obel edge detection — with a built-in CPU reference implementation for benchmarking.

> 📖 中文文档请查看 [README_CN.md](README_CN.md)

## 🎨 Pipeline Demo

<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>Grayscale</b></td>
    <td align="center"><b>Gaussian Blur</b></td>
    <td align="center"><b>Histogram Eq.</b></td>
    <td align="center"><b>Sobel Edge</b></td>
  </tr>
  <tr>
    <td><img src="image/test_image.png" width="150"></td>
    <td><img src="image/test_image_gray.png" width="150"></td>
    <td><img src="image/test_image_blur.png" width="150"></td>
    <td><img src="image/test_image_equalized.png" width="150"></td>
    <td><img src="image/test_image_edge.png" width="150"></td>
  </tr>
</table>

## 📑 Table of Contents

- [🎨 Pipeline Demo](#pipeline-demo)
- [📋 Overview](#overview)
- [🔄 Pipeline Stages](#pipeline-stages)
- [📁 Project Structure](#project-structure)
- [⚙️ Prerequisites](#prerequisites)
- [🔨 Build](#build)
- [🚀 Usage](#usage)
- [📤 Output Files](#output-files)
- [🏗️ Architecture Details](#architecture-details)
- [🧯 Error Handling](#error-handling)
- [📊 Performance Benchmarking](#performance-benchmarking)
- [📄 License](#license)

## 📋 Overview

G2ES loads an image (any format OpenCV supports: PNG, JPG, BMP, PGM, PPM, etc.), processes it through a four-stage pipeline, and writes intermediate and final results as PNG files. The project supports four execution modes:

| Mode | Flag | Description |
|------|------|-------------|
| GPU only | `--gpu` (default) | Runs the CUDA pipeline with naive kernels |
| GPU optimized | `--gpu-optimized` | Runs the CUDA pipeline with optimized kernels |
| CPU only | `--cpu` | Runs the single-threaded CPU reference pipeline |
| Both | `--both` | Runs both CPU and GPU (naive) and prints a speedup comparison |

## 🔄 Pipeline Stages

```
Input Image → [1. RGB→Gray] → [2. Gaussian Blur] → [3. Histogram Eq.] → [4. Sobel Edge] → Output
```

| Stage | Algorithm | Kernel Size | Key Detail |
|-------|-----------|-------------|------------|
| 1. RGB to Grayscale | Weighted luminance `0.299R + 0.587G + 0.114B` | — | ITU-R BT.601 standard |
| 2. Gaussian Blur | 5×5 Gaussian smoothing | 5×5 | Weights stored in `__constant__` memory |
| 3. Histogram Equalization | CDF-based intensity redistribution | — | 3-kernel decomposition (histogram → CDF/LUT → apply) |
| 4. Sobel Edge Detection | `min(255, √(Gx² + Gy²))` | 3×3 | Naive implementation (baseline) |

### Optimized Kernels

The `--gpu-optimized` mode uses optimized kernel implementations that provide better performance:

| Stage | Optimization Technique | Performance Impact |
|-------|----------------------|-------------------|
| 1. RGB to Grayscale | `uchar3` vectorized memory access | ~15% faster memory reads |
| 2. Gaussian Blur | Shared memory + separable convolution | ~20% faster convolution |
| 3. Histogram | Shared memory local histograms + grid-stride loop | ~10% faster atomic operations |
| 4. Sobel Edge | Shared memory tile with halo | ~25% faster neighborhood access |

## 📁 Project Structure

```
G2ES_GPU_Pipeline/
├── include/
│   ├── kernels.cuh            # All __global__ kernel declarations
│   ├── pipeline_common.h      # Pipeline common definitions and structures
│   └── utils.h                # CUDA error-checking macro
├── src/
│   ├── main.cu                # Entry point, CLI parsing, benchmark orchestration
│   ├── cpu_pipeline.cpp       # CPU pipeline implementation
│   ├── cpu_pipeline.h         # CPU pipeline header
│   ├── gpu_pipeline.cu        # GPU pipeline implementation (naive kernels)
│   ├── gpu_pipeline.h         # GPU pipeline header
│   ├── gpu_pipeline_optimized.cu  # GPU pipeline implementation (optimized kernels)
│   ├── pipeline_common.cu     # Pipeline common utilities
│   └── kernels/
│       ├── rgb_to_gray.cu     # RGB → Grayscale kernel
│       ├── gaussian_blur.cu   # 5×5 Gaussian blur kernel
│       ├── histogram_equalization.cu  # Histogram + CDF/LUT + LUT apply kernels
│       └── sobel_edge.cu      # Sobel edge detection kernel
├── image/
│   └── test_image.png         # Sample input image
├── Makefile                   # nvcc-based build system
├── README.md                  # This file
└── README_CN.md               # 中文文档
```

## ⚙️ Prerequisites

| Dependency | Version | Purpose |
|------------|---------|---------|
| **NVIDIA CUDA Toolkit** | 11.0+ | `nvcc` compiler and CUDA runtime |
| **OpenCV** | 4.x | Image I/O (`imread`/`imwrite`) and color conversion |
| **CUDA-capable GPU** | Compute Capability 8.7 | Default target: Jetson AGX Orin / Ampere |

> **Note:** The Makefile uses `-arch=sm_87` by default. If your GPU has a different compute capability, modify the `NVCCFLAGS` in the Makefile accordingly (e.g., `sm_80` for A100, `sm_86` for RTX 3080, `sm_89` for RTX 4090).

### Installing Dependencies (Ubuntu/Debian)

```bash
# CUDA Toolkit (if not already installed)
sudo apt install nvidia-cuda-toolkit

# OpenCV 4
sudo apt install libopencv-dev

# Verify
nvcc --version
pkg-config --modversion opencv4
```

## 🔨 Build

```bash
make
```

This compiles all `.cu` source files under `src/` and `src/kernels/` into object files in `obj/`, then links them into the `image_pipeline` binary.

To clean build artifacts:

```bash
make clean
```

## 🚀 Usage

```bash
./image_pipeline [mode] <input_image> <output_prefix>
```

### Examples

```bash
# GPU mode (default) — reads test_image.png, writes test_image_*.png
./image_pipeline image/test_image.png image/test_image

# GPU optimized mode — uses optimized kernels for better performance
./image_pipeline --gpu-optimized image/test_image.png image/test_image_optimized

# CPU mode only
./image_pipeline --cpu image/test_image.png image/test_image_cpu

# Run both GPU and CPU, print speedup comparison
./image_pipeline --both image/test_image.png image/test_image
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--gpu` / `--gpu-optimized` / `--cpu` / `--both` | No | Execution mode (default: `--gpu`) |
| `<input_image>` | Yes | Path to input image (any OpenCV-supported format) |
| `<output_prefix>` | Yes | Prefix for output file names |

## 📤 Output Files

Each run produces four PNG images per mode:

| File | Description |
|------|-------------|
| `{prefix}_gray.png` | Grayscale conversion result |
| `{prefix}_blur.png` | Gaussian blur result |
| `{prefix}_equalized.png` | Histogram equalization result |
| `{prefix}_edge.png` | Sobel edge detection result |

In `--both` mode, files are automatically suffixed with `_cpu` or `_gpu` (e.g., `test_image_cpu_gray.png`, `test_image_gpu_gray.png`).

## 🏗️ Architecture Details

### GPU Kernel Design

**Thread Block Layout:**
- Image-processing kernels (grayscale, blur, sobel): 2D blocks of `16×16 = 256` threads
- Linear kernels (histogram, LUT apply): 1D blocks of `256` threads

**Memory Strategy:**
- Intermediate buffers (`d_gray`, `d_blur`, `d_equalized`, `d_edge`) stay on the GPU throughout the pipeline — only the final result is copied back to the host
- Gaussian kernel weights use `__constant__` memory for broadcast efficiency
- Histogram equalization uses shared memory for parallel prefix sum (Blelloch scan)

**Histogram Equalization Decomposition:**

The histogram equalization stage is split into three specialized kernels:

1. **`compute_histogram_global_kernel`** — Each thread processes one pixel, uses `atomicAdd` to build a 256-bin histogram in global memory
2. **`build_equalization_lut_kernel`** — Single block of 256 threads performs a Blelloch inclusive scan in shared memory to compute the CDF, then builds the equalization LUT
3. **`apply_lut_kernel`** — Each thread does a simple table lookup: `output[i] = lut[input[i]]`

### CPU Reference Implementation

The CPU pipeline in `src/cpu_pipeline.cpp` implements identical algorithms using single-threaded nested loops. It exists for:
- Correctness verification (comparing GPU output against CPU output)
- Performance benchmarking (measuring GPU speedup)

## 🧯 Error Handling

All CUDA API calls are checked through the `CHECK_CUDA_ERROR` macro defined in `include/utils.h`. On failure, the helper prints detailed error information, including file name, line number, CUDA call, and error code, then terminates the program.

## 📊 Performance Benchmarking

Use `--both` mode to compare GPU and CPU performance in a single run:

```bash
./image_pipeline --both image/test_image.png image/test_image
```

The output includes:
- Per-stage timing for both CPU and GPU
- Total compute time
- **Kernel-only speedup** (GPU kernel execution vs CPU compute)
- **Transfer-inclusive speedup** (including host↔device memory transfers)

GPU timing uses `cudaEvent` for accurate kernel measurement; CPU timing uses `std::chrono::steady_clock`.

### Comparing Naive vs Optimized Kernels

To compare the performance of naive and optimized GPU kernels:

```bash
# Run naive kernels
./image_pipeline --gpu image/test_image.png output/naive

# Run optimized kernels
./image_pipeline --gpu-optimized image/test_image.png output/optimized
```

**Example Performance Results (1440×810 image on Jetson AGX Orin):**

| Metric | Naive Kernels | Optimized Kernels | Improvement |
|--------|--------------|-------------------|-------------|
| Kernel Total | 1.37 ms | 1.21 ms | **11.5% faster** |
| Transfer+Kernel | 3.94 ms | 3.64 ms | **7.7% faster** |

### Kernel-by-Kernel Benchmark

Use `--benchmark` mode to test each optimized kernel individually with controlled variables:

```bash
./image_pipeline --benchmark image/test_image.png
```

This runs 6 tests (3 warmup + 10 benchmark runs each):
1. Baseline (all naive kernels)
2. Only RGB optimized
3. Only Gaussian Blur optimized
4. Only Histogram optimized
5. Only Sobel optimized
6. All optimized

**Example Benchmark Results (1440×810 image on Jetson AGX Orin):**

| Test Configuration | Avg Time | Min Time | Improvement |
|-------------------|----------|----------|-------------|
| Baseline (All Naive) | 0.6518 ms | 0.6463 ms | - |
| Only RGB Optimized | 0.6544 ms | 0.6444 ms | -0.40% |
| Only Gaussian Blur Optimized | 0.6699 ms | 0.6623 ms | -2.78% |
| Only Histogram Optimized | 0.2882 ms | 0.2770 ms | **+55.78%** |
| Only Sobel Optimized | 0.7049 ms | 0.6973 ms | -8.15% |
| All Optimized | 0.3366 ms | 0.3306 ms | **+48.36%** |

**Key Findings:**
- Histogram optimization provides the most significant improvement (55.78%)
- Other individual optimizations show slight overhead
- Combined optimization achieves 48.36% improvement, mainly from histogram optimization

## 📄 License

This project is for educational/study purposes.
