#include "pipeline_common.h"

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name
              << " [--gpu|--cpu|--both] [input_image] [output_prefix]" << std::endl;
    std::cerr << "Default mode:  --gpu" << std::endl;
    std::cerr << "Default input:  " << kDefaultInputPath << std::endl;
    std::cerr << "Default output: output/" << std::endl;
}

bool parse_mode_flag(const std::string& value, RunMode* mode) {
    if (value == "--gpu") {
        *mode = RunMode::kGpu;
        return true;
    }
    if (value == "--cpu") {
        *mode = RunMode::kCpu;
        return true;
    }
    if (value == "--both") {
        *mode = RunMode::kBoth;
        return true;
    }
    return false;
}

Options parse_options(int argc, char** argv) {
    Options options;
    int arg_index = 1;

    if (argc > 1) {
        const std::string first_arg = argv[1];
        if (first_arg == "--help" || first_arg == "-h") {
            options.show_help = true;
            return options;
        }
        if (parse_mode_flag(first_arg, &options.mode)) {
            arg_index = 2;
        } else if (first_arg.rfind("--", 0) == 0) {
            throw std::runtime_error("Unknown option: " + first_arg);
        }
    }

    const int remaining_args = argc - arg_index;
    if (remaining_args > 2) {
        throw std::runtime_error("Too many arguments.");
    }

    if (remaining_args >= 1) {
        options.input_path = argv[arg_index];
    }

    options.output_prefix_provided = (remaining_args == 2);
    if (options.output_prefix_provided) {
        options.output_prefix = argv[arg_index + 1];
    } else {
        const std::string stripped = strip_extension(options.input_path);
        const size_t slash = stripped.find_last_of("/\\");
        const std::string basename = (slash == std::string::npos)
                                         ? stripped
                                         : stripped.substr(slash + 1);
        options.output_prefix = "output/" + basename;
    }
    return options;
}

ImageData load_image(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        throw std::runtime_error("Failed to open input image: " + path);
    }
    if (raw.depth() != CV_8U) {
        throw std::runtime_error("Only 8-bit input images are supported.");
    }

    ImageData image;
    image.width = raw.cols;
    image.height = raw.rows;

    cv::Mat normalized;
    if (raw.channels() == 1) {
        normalized = raw;
        image.channels = 1;
    } else if (raw.channels() == 3) {
        cv::cvtColor(raw, normalized, cv::COLOR_BGR2RGB);
        image.channels = 3;
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, normalized, cv::COLOR_BGRA2RGB);
        image.channels = 3;
    } else {
        throw std::runtime_error("Only grayscale, RGB, or RGBA images are supported.");
    }

    if (!normalized.isContinuous()) {
        normalized = normalized.clone();
    }

    const size_t bytes = static_cast<size_t>(image.width) *
                         static_cast<size_t>(image.height) *
                         static_cast<size_t>(image.channels);
    image.pixels.assign(normalized.data, normalized.data + bytes);
    return image;
}

void write_png(const std::string& path,
               const std::vector<unsigned char>& pixels,
               const int width,
               const int height) {
    cv::Mat image(height, width, CV_8UC1, const_cast<unsigned char*>(pixels.data()));
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to write output image: " + path);
    }
}

double write_pipeline_outputs(const std::string& output_prefix,
                              const PipelineOutputs& outputs,
                              const int width,
                              const int height) {
    const auto write_start = Clock::now();
    write_png(output_prefix + "_gray.png", outputs.gray, width, height);
    write_png(output_prefix + "_blur.png", outputs.blur, width, height);
    write_png(output_prefix + "_equalized.png", outputs.equalized, width, height);
    write_png(output_prefix + "_edge.png", outputs.edge, width, height);
    return elapsed_ms(write_start, Clock::now());
}

PipelineOutputs make_outputs(const int pixels) {
    PipelineOutputs outputs;
    outputs.gray.resize(pixels);
    outputs.blur.resize(pixels);
    outputs.equalized.resize(pixels);
    outputs.edge.resize(pixels);
    return outputs;
}

void print_cpu_info() {
    std::cout << "=== CPU Device Info ===" << std::endl;

    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::string model_name;
    std::string cpu_mhz;
    std::string bogo_mips;
    int cpu_cores = 0;

    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                model_name = line.substr(colon + 2);
            }
        } else if (line.find("cpu MHz") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                cpu_mhz = line.substr(colon + 2);
            }
        } else if (line.find("BogoMIPS") != std::string::npos && bogo_mips.empty()) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                bogo_mips = line.substr(colon + 2);
            }
        } else if (line.find("cpu cores") != std::string::npos) {
            size_t colon = line.find(':');
            if (colon != std::string::npos) {
                std::istringstream iss(line.substr(colon + 2));
                iss >> cpu_cores;
            }
        }
    }

    if (cpu_cores == 0) {
        cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
    }

    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    long total_mem_mb = (pages * page_size) / (1024 * 1024);

    std::string max_freq;
    std::ifstream freq_file("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
    if (freq_file.is_open()) {
        long freq_khz = 0;
        freq_file >> freq_khz;
        if (freq_khz > 0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << (freq_khz / 1000.0);
            max_freq = oss.str();
        }
    }

    if (!model_name.empty()) {
        std::cout << "Model:           " << model_name << std::endl;
    }
    if (!cpu_mhz.empty()) {
        std::cout << "CPU MHz:         " << cpu_mhz << " MHz" << std::endl;
    } else if (!max_freq.empty()) {
        std::cout << "CPU Max Freq:    " << max_freq << " MHz" << std::endl;
    }
    if (!bogo_mips.empty()) {
        std::cout << "BogoMIPS:        " << bogo_mips << std::endl;
    }
    std::cout << "CPU Cores:       " << cpu_cores << std::endl;
    std::cout << "System Memory:   " << total_mem_mb << " MB" << std::endl;
    std::cout << "========================\n" << std::endl;
}
