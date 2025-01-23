#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <chrono> // For time measurement
#include <sys/resource.h> // For memory and resource usage on Linux
#include <stdexcept>
#include <numeric>  // For std::inner_product

// Function to load scaler parameters from TXT file
std::pair<std::vector<float>, std::vector<float>> load_scaler_params(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open scaler parameters file: " + filename);
    }

    std::string line;
    std::vector<float> mean, std;

    // Read mean values
    std::getline(file, line); // Skip the "Mean Values:"
    std::getline(file, line);
    std::stringstream ss_mean(line);
    float value;
    while (ss_mean >> value) {
        mean.push_back(value);
    }

    // Read standard deviation values
    std::getline(file, line); // Skip the "Standard Deviation Values:"
    std::getline(file, line);
    std::stringstream ss_std(line);
    while (ss_std >> value) {
        std.push_back(value);
    }

    if (mean.empty() || std.empty())
        throw std::runtime_error("Error: Failed to parse mean and std from the file");

    return std::make_pair(mean, std);
}

// Function to normalize input vector
std::vector<float> normalize_input(const std::vector<float>& input, const std::vector<float>& mean, const std::vector<float>& std) {
    if (input.size() != mean.size() || input.size() != std.size()) {
        throw std::runtime_error("Input vector size does not match mean or std vector sizes.");
    }
    std::vector<float> normalized_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        normalized_input[i] = (input[i] - mean[i]) / std[i];
    }
    return normalized_input;
}

int main() {
    // Load ONNX model
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "SOH_best_model.onnx", session_options);

    // Input and output names (defined statically)
    const char* input_names[] = {"float_input"};  // Input name as obtained from the ONNX model
    const char* output_names[] = {"variable"};


   // Load scaler parameters
    std::vector<float> mean, std_dev;
    try {
        std::tie(mean, std_dev) = load_scaler_params("scaler_params.txt");
    } catch (const std::runtime_error& e) {
        std::cerr << "Error loading scaler parameters: " << e.what() << std::endl;
        return 1;
    }

    // Test input data
    std::vector<float> test_input = {3.443567, -2.010116, 34.197394, 1.998000, 2.476000, 1733.859000, 1.485868, 100.000000};

    // Normalize input
    std::vector<float> normalized_input = normalize_input(test_input, mean, std_dev);

    // Define input tensor
    std::vector<int64_t> input_dims = {1, (int64_t)normalized_input.size()};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, normalized_input.data(), normalized_input.size(), input_dims.data(), input_dims.size());

    // Start time measurement
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run inference
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));

    Ort::RunOptions run_options;
    std::vector<Ort::Value> output_tensors = session.Run(
        run_options,
        input_names, ort_inputs.data(), 1,
        output_names,  1
       );


     // Stop time measurement
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Get output data
    float prediction = output_tensors[0].GetTensorMutableData<float>()[0];


     // Memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    long memory_usage_kb = usage.ru_maxrss;


    // Display results
    std::cout << "Predicted SOH: " << prediction << std::endl;
    std::cout << "Inference Time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Memory Usage: " << memory_usage_kb << " KB" << std::endl;

    return 0;
}