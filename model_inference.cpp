// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <sstream>
// #include <cmath>

// // Function to load scaler parameters from a file
// void load_scaler_params(const std::string& scaler_path, std::vector<float>& means, std::vector<float>& scales) {
//     std::ifstream file(scaler_path);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open scaler file.\n";
//         exit(1);
//     }

//     std::string line;
//     // Load means
//     std::getline(file, line);
//     std::stringstream ss(line);
//     float mean;
//     while (ss >> mean) {
//         means.push_back(mean);
//         if (ss.peek() == ',') ss.ignore();
//     }

//     // Load scales
//     std::getline(file, line);
//     std::stringstream ss2(line);
//     float scale;
//     while (ss2 >> scale) {
//         scales.push_back(scale);
//         if (ss2.peek() == ',') ss2.ignore();
//     }
// }

// // Function to scale input data
// void scale_input(float* input, const std::vector<float>& means, const std::vector<float>& scales, int size) {
//     for (int i = 0; i < size; i++) {
//         input[i] = (input[i] - means[i]) / scales[i];
//     }
// }

// // Dummy predict function for illustration
// float predict(const std::vector<float>& input) {
//     // Placeholder for actual prediction logic
//     // For demonstration, return a random value based on input
//     return input[0] * 0.5; // Change this to actual model logic
// }

// int main() {
//     // Load scaler parameters
//     std::vector<float> means;
//     std::vector<float> scales;
//     load_scaler_params("scaler.txt", means, scales);

//     // Example input data
//     float input_data[7] = {5.0, 131.0, 103.0, 2830.0, 15.9, 78.0, 2.0};

//     // Scale input data
//     scale_input(input_data, means, scales, 7);

//     // Predict MPG
//     float predicted_mpg = predict(std::vector<float>(input_data, input_data + 7));
//     std::cout << "Predicted MPG: " << predicted_mpg << std::endl;

//     return 0;
// }


// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <sstream>
// #include <onnxruntime_cxx_api.h>

// // Function to load scaler parameters from a file
// void load_scaler_params(const std::string& scaler_path, std::vector<float>& means, std::vector<float>& scales) {
//     std::ifstream file(scaler_path);
//     if (!file.is_open()) {
//         std::cerr << "Failed to open scaler file.\n";
//         exit(1);
//     }

//     std::string line;
//     // Load means
//     std::getline(file, line);
//     std::stringstream ss(line);
//     float mean;
//     while (ss >> mean) {
//         means.push_back(mean);
//         if (ss.peek() == ',') ss.ignore();
//     }

//     // Load scales
//     std::getline(file, line);
//     std::stringstream ss2(line);
//     float scale;
//     while (ss2 >> scale) {
//         scales.push_back(scale);
//         if (ss2.peek() == ',') ss2.ignore();
//     }
// }

// // Function to scale input data
// void scale_input(float* input, const std::vector<float>& means, const std::vector<float>& scales, int size) {
//     for (int i = 0; i < size; i++) {
//         input[i] = (input[i] - means[i]) / scales[i];
//     }
// }

// // Function to perform prediction using ONNX model
// float predict(const std::vector<float>& input) {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelInference");
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
    
//     // Set the optimization level (optional)
//     // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
//     Ort::Session session(env, "best_model.onnx", session_options);

//     // Prepare input tensor
//     std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input.size())}; // Batch size of 1
//     std::vector<float> input_data = input;

//     // Create input tensor
//     Ort::AllocatorWithDefaultOptions allocator; // Create an allocator instance
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

//     // Specify input and output names
//     const char* input_names[] = {"input"}; // Replace "input" with your actual input node name
//     const char* output_names[] = {"output"}; // Assuming the output node name is "output"

//     // Run inference
//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // Retrieve the output value
//     float* output_arr = output_tensors[0].GetTensorMutableData<float>();
//     return output_arr[0]; // Assuming single output
// }

// int main() {
//     // Load scaler parameters
//     std::vector<float> means;
//     std::vector<float> scales;
//     load_scaler_params("scaler.txt", means, scales);

//     // Example input data
//     float input_data[7] = {5.0, 131.0, 103.0, 2830.0, 15.9, 78.0, 2.0};

//     // Scale input data
//     scale_input(input_data, means, scales, 7);

//     // Predict MPG
//     float predicted_mpg = predict(std::vector<float>(input_data, input_data + 7));
//     std::cout << "Predicted MPG: " << predicted_mpg << std::endl;

//     return 0;
// }



#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <onnxruntime_cxx_api.h>

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

// Function to perform inference using ONNX Runtime
float predict(const std::vector<float>& input_data, Ort::Session& session) {
    // Prepare the input shape and names
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())}; // Batch size of 1
    // Define input and output names
    const char* input_names[] = {"float_input"};  // Input name as obtained from the ONNX model
    const char* output_names[] = {"variable"};     // Output name as obtained from the ONNX model


    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Cast to non-const float* for input_data
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

    // Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Get the output
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();
    return output_arr[0]; // Assuming single output
}

int main() {
    // Load scaler parameters
    std::vector<float> means;
    std::vector<float> scales;
    load_scaler_params("scaler.txt", means, scales);

    // Example input data
    float input_data[7] = {5.0, 131.0, 103.0, 2830.0, 15.9, 78.0, 2.0};

    // Scale input data
    scale_input(input_data, means, scales, 7);

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "model_inference");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "best_model.onnx", session_options); // Make sure the model file is in the same directory

    // Predict MPG
    float predicted_mpg = predict(std::vector<float>(input_data, input_data + 7), session);
    std::cout << "Predicted MPG: " << predicted_mpg << std::endl;

    return 0;
}
