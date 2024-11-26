

// #include <onnxruntime_cxx_api.h>
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <algorithm>  // For std::max_element

// // Function to perform inference using ONNX Runtime
// int predict(const std::vector<float>& input_data, Ort::Session& session) {
//     // Prepare the input shape (batch size = 1, feature size = 6)
//     std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};  // Shape = {1, 6}

//     // Create input tensor (ensure that data is of type float and properly shaped)
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

//     // Create tensor with float type since the model expects float
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

//     // Debug: Print the input tensor shape and size
//     std::cout << "Input tensor shape: ";
//     for (size_t i = 0; i < input_shape.size(); ++i) {
//         std::cout << input_shape[i] << " ";
//     }
//     std::cout << std::endl;

//     // Get input/output names (use the correct names from your model)
//     const char* input_name = "input";  // Updated input name
//     const char* output_name = "output_label";  // Updated output name

//     // Run the model
//     auto output_tensors = session.Run(Ort::RunOptions{nullptr},
//                                       &input_name, &input_tensor, 1,  // Inputs
//                                       &output_name, 1);               // Output (predicted label)

//     // Get the output tensor data (assuming output is of type float)
//     float* output_arr = output_tensors[0].GetTensorMutableData<float>();

//     // Find the predicted class (index of the highest value in output)
//     int predicted_class = std::distance(output_arr, std::max_element(output_arr, output_arr + 4));  // Assuming 4 output classes

//     return predicted_class;  // Return the predicted class (0, 1, 2, or 3)
// }

// int main() {
//     std::vector<float> input_data(6);  // Input data of type float (assuming the model expects float)

//     // Get input data from the user
//     std::cout << "Please enter 6 input values separated by spaces: ";
//     for (int i = 0; i < 6; ++i) {
//         std::cin >> input_data[i];  // Input values as float
//     }

//     // Initialize ONNX Runtime environment and session
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "classification");
//     Ort::SessionOptions session_options;
//     Ort::Session session(env, "RF.onnx", session_options);  // Ensure model path is correct DT.onnx

//     // Perform prediction
//     try {
//         int predicted_class = predict(input_data, session);
//         std::cout << "Predicted Class: " << predicted_class << std::endl;
//     } catch (const Ort::Exception& e) {
//         std::cerr << "Error during inference: " << e.what() << std::endl;
//     }

//     return 0;
// }


#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>  // For std::max_element
#include <sys/resource.h>  // For memory usage (Linux-specific)

// Function to get current memory usage in MB
double getMemoryUsageMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;  // Convert from KB to MB
}

// Function to perform inference using ONNX Runtime
int predict(const std::vector<float>& input_data, Ort::Session& session) {
    // Prepare the input shape (batch size = 1, feature size = 6)
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};  // Shape = {1, 6}

    // Create input tensor (ensure that data is of type float and properly shaped)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

    // Debug: Print the input tensor shape
    std::cout << "Input tensor shape: ";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << " ";
    }
    std::cout << std::endl;

    // Get input/output names (use the correct names from your model)
    const char* input_name = "dense_68_input";  // Updated input name
    const char* output_name = "dense_71";  // Updated output name

    // Measure inference time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      &input_name, &input_tensor, 1,  // Inputs
                                      &output_name, 1);               // Outputs

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> inference_time = end_time - start_time;

    // Print inference time
    std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;

    // Get the output tensor data (assuming output is of type float)
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();

    // Find the predicted class (index of the highest value in output)
    int predicted_class = std::distance(output_arr, std::max_element(output_arr, output_arr + 4));  // Assuming 4 output classes

    return predicted_class;  // Return the predicted class
}

int main() {
    std::vector<float> input_data(6);  // Input data of type float (assuming the model expects float)

    // Get input data from the user
    std::cout << "Please enter 6 input values separated by spaces (4 1 3 3 3 2): ";
    for (int i = 0; i < 6; ++i) {
        std::cin >> input_data[i];  // Input values as float
    }

    // Initialize ONNX Runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "classification1");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "classification_ann_model.onnx", session_options);  // Ensure model path is correct

    // Measure memory usage before inference
    double memory_before = getMemoryUsageMB();

    // Perform prediction
    try {
        int predicted_class = predict(input_data, session);
        std::cout << "Predicted Class: " << predicted_class+1 << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

    // Measure memory usage after inference
    double memory_after = getMemoryUsageMB();
    std::cout << "Memory usage: " << memory_after << " MB" << std::endl;

    return 0;
}
