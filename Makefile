NVCC = nvcc
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4) -I/usr/include/opencv4
OPENCV_LIBS := $(shell pkg-config --libs opencv4)
NVCCFLAGS = -O3 -std=c++17 -Iinclude $(OPENCV_CFLAGS) -arch=sm_87 # Jetson AGX Orin / Ampere

SRC_DIR = src
KERNEL_DIR = src/kernels
OBJ_DIR = obj
BIN = image_pipeline

SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(KERNEL_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(BIN) cpu_benchmark

.PHONY: all clean
