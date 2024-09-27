#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <onnxruntime_cxx_api.h>
#include <chrono> // For time measurement
#include <sys/resource.h> // For memory and resource usage on Linux

// Function to load scaler parameters from a file
void load_scaler_params(const std::string& scaler_path, std::vector<float>& means, std::vector<float>& scales) {
    std::ifstream file(scaler_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open scaler file.\n";
        exit(1);
    }

    std::string line;
    // Load means
    std::getline(file, line);
    std::stringstream ss(line);
    float mean;
    while (ss >> mean) {
        means.push_back(mean);
        if (ss.peek() == ',') ss.ignore();
    }

    // Load scales
    std::getline(file, line);
    std::stringstream ss2(line);
    float scale;
    while (ss2 >> scale) {
        scales.push_back(scale);
        if (ss2.peek() == ',') ss2.ignore();
    }
}

// Function to scale input data
void scale_input(float* input, const std::vector<float>& means, const std::vector<float>& scales, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = (input[i] - means[i]) / scales[i];
    }
}

// Function to get current memory usage (resident set size)
long getMemoryUsageInKB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Max resident set size in kilobytes
}

// Function to perform inference using ONNX Runtime and measure time and memory
float predict(const std::vector<float>& input_data, Ort::Session& session) {
    // Prepare the input shape and names
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())}; // Batch size of 1
    const char* input_names[] = {"float_input"};  // Input name as obtained from the ONNX model
    const char* output_names[] = {"variable"};     // Output name as obtained from the ONNX model

    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

    // Start measuring inference time
    auto start = std::chrono::high_resolution_clock::now();
    long memoryBefore = getMemoryUsageInKB(); // Get memory usage before inference

    // Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // End measuring inference time
    auto end = std::chrono::high_resolution_clock::now();
    long memoryAfter = getMemoryUsageInKB(); // Get memory usage after inference

    std::chrono::duration<double, std::milli> inference_time = end - start;
    long memoryUsed = memoryAfter - memoryBefore; // Calculate memory used during inference

    // Print inference time and memory usage
    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
    std::cout << "Memory used during inference: " << memoryUsed << " KB" << std::endl;

    // Get the output
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();
    return output_arr[0]; // Assuming single output
}

int main() {
    // Load scaler parameters
    std::vector<float> means;
    std::vector<float> scales;
    load_scaler_params("scaler.txt", means, scales);

    // Create a vector to store user input data
    std::vector<float> input_data(7);

    // Get input data from the user
    std::cout << "Please enter 7 input values separated by spaces (e.g., 5.0 131.0 103.0 2830.0 15.9 78.0 2.0): ";
    for (int i = 0; i < 7; ++i) {
        std::cin >> input_data[i];
    }

    // Scale input data
    scale_input(input_data.data(), means, scales, 7);

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "best_model.onnx", session_options); // Make sure the model file is in the same directory

    // Predict MPG and measure inference time and memory
    float predicted_mpg = predict(input_data, session);
    std::cout << "Predicted MPG: " << predicted_mpg << std::endl;

    return 0;
}
