NVCC = nvcc
CXX = g++
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4) -I/usr/include/opencv4
OPENCV_LIBS := $(shell pkg-config --libs opencv4)
NVCCFLAGS = -O3 -std=c++17 -Iinclude $(OPENCV_CFLAGS) -arch=sm_87 # Jetson AGX Orin / Ampere
CXXFLAGS = -O3 -std=c++17 -Iinclude $(OPENCV_CFLAGS)

SRC_DIR = src
KERNEL_DIR = src/kernels
UTILS_DIR = src/utils
OBJ_DIR = obj
BIN = image_pipeline

CU_SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(KERNEL_DIR)/*.cu) $(wildcard $(UTILS_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
CU_OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SRCS))
OBJS = $(CU_OBJS) $(CPP_OBJS)

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJ_DIR) $(BIN) cpu_benchmark

.PHONY: all clean
