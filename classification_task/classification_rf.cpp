// #include <onnxruntime_cxx_api.h>
// #include <iostream>
// #include <vector>
// #include <chrono> // For time measurement
// #include <sys/resource.h> // For memory measurement

// // Function to get current memory usage
// size_t getCurrentMemoryUsage() {
//     struct rusage usage;
//     getrusage(RUSAGE_SELF, &usage);
//     return usage.ru_maxrss; // Return memory in kilobytes
// }

// int main() {
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceExample");

//     // Create session options
//     Ort::SessionOptions session_options;
//     session_options.SetIntraOpNumThreads(1);
//     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

//     // Load the ONNX model
//     const char* model_path = "ML_classification.onnx";
//     Ort::Session session(env, model_path, session_options);
//     std::cout << "Model loaded successfully!" << std::endl;

//     // Input and Output names
//     Ort::AllocatorWithDefaultOptions allocator;

//     auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
//     const char* input_name = input_name_ptr.get();
//     std::cout << "Input name: " << input_name << std::endl;

//     auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);
//     const char* output_name = output_name_ptr.get();
//     std::cout << "Output name: " << output_name << std::endl;

//     // Take user input for features
//     float buying, maint, doors, persons, lug_boot, safety;
//     std::cout << "Enter the values for the following features:\n";
//     std::cout << "Buying (1-4): ";
//     std::cin >> buying;
//     std::cout << "Maint (1-4): ";
//     std::cin >> maint;
//     std::cout << "Doors (1-4): ";
//     std::cin >> doors;
//     std::cout << "Persons (1-3): ";
//     std::cin >> persons;
//     std::cout << "Lug_boot (1-3): ";
//     std::cin >> lug_boot;
//     std::cout << "Safety (1-3): ";
//     std::cin >> safety;

//     // Prepare input data
//     std::vector<float> input_tensor_values = {buying, maint, doors, persons, lug_boot, safety};
//     std::vector<int64_t> input_tensor_shape = {1, 6}; // Batch size = 1, features = 6

//     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
//         memory_info, input_tensor_values.data(), input_tensor_values.size(),
//         input_tensor_shape.data(), input_tensor_shape.size());

//     // Measure inference time
//     auto start_time = std::chrono::high_resolution_clock::now();
//     size_t start_memory = getCurrentMemoryUsage();

//     // Run inference
//     std::vector<const char*> input_names = {input_name};
//     std::vector<const char*> output_names = {output_name};

//     auto output_tensors = session.Run(
//         Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
//         output_names.data(), 1);

//     auto end_time = std::chrono::high_resolution_clock::now();
//     size_t end_memory = getCurrentMemoryUsage();

//     // Calculate inference time
//     auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

//     // Get the output data
//     float* floatarr = output_tensors[0].GetTensorMutableData<float>();

//     std::cout << "Inference result: ";
//     for (size_t i = 0; i < output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(); i++) {
//         std::cout << floatarr[i] << " ";
//     }
//     std::cout << std::endl;

//     // Output performance metrics
//     std::cout << "Inference time: " << inference_time << " microseconds" << std::endl;
//     std::cout << "Memory usage: " << (end_memory - start_memory) << " KB" << std::endl;

//     return 0;
// }







// #include <iostream>
// #include <vector>
// #include <onnxruntime_cxx_api.h>
// #include <chrono> // For time measurement
// #include <sys/resource.h> // For memory and resource usage on Linux
// #include <algorithm> // For std::max_element

// // Function to get current memory usage (resident set size)
// long getMemoryUsageInKB() {
//     struct rusage usage;
//     getrusage(RUSAGE_SELF, &usage);
//     return usage.ru_maxrss; // Max resident set size in kilobytes
// }

// // Function to perform inference using ONNX Runtime and measure time and memory
// int predict(const std::vector<float>& input_data, Ort::Session& session) {
//     // Prepare the input shape and names
//     std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())}; // Batch size of 1
//     const char* input_names[] = {"float_input"};  // Input name as obtained from the ONNX model
//     const char* output_names[] = {"output_label"}; // Output name as obtained from the ONNX model

//     // Create input tensor
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

//     // Start measuring inference time
//     auto start = std::chrono::high_resolution_clock::now();
//     long memoryBefore = getMemoryUsageInKB(); // Get memory usage before inference

//     // Run the model
//     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

//     // End measuring inference time
//     auto end = std::chrono::high_resolution_clock::now();
//     long memoryAfter = getMemoryUsageInKB(); // Get memory usage after inference

//     std::chrono::duration<double, std::milli> inference_time = end - start;
//     long memoryUsed = memoryAfter - memoryBefore; // Calculate memory used during inference

//     // Print inference time and memory usage
//     std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
//     std::cout << "Memory used during inference: " << memoryUsed << " KB" << std::endl;

//     // Get the output
//     float* output_arr = output_tensors[0].GetTensorMutableData<float>();

//     // Debug: Output the raw result
//     std::cout << "Raw Output: ";
//     for (size_t i = 0; i < 4; ++i) {  // Assuming 4 classes
//         std::cout << output_arr[i] << " ";
//     }
//     std::cout << std::endl;

//     // Find the predicted class (index of the highest value in output)
//     int predicted_class = std::distance(output_arr, std::max_element(output_arr, output_arr + 4)); // For 4 classes (0, 1, 2, 3)

//     return predicted_class; // Return the predicted class (0, 1, 2, or 3)
// }

// int main() {
//     // Create a vector to store user input data
//     std::vector<float> input_data(6);  // 6 input values expected

//     // Get input data from the user
//     std::cout << "Please enter 6 input values separated by spaces (e.g.,3 2 1 2 1 3): ";
//     for (int i = 0; i < 6; ++i) {
//         std::cin >> input_data[i];
//     }

//     // Initialize ONNX Runtime
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "classification");
//     Ort::SessionOptions session_options;
//     Ort::Session session(env, "ML_classification.onnx", session_options); // Make sure the model file is in the same directory

//     // Predict the class and measure inference time and memory
//     int predicted_class = predict(input_data, session);
//     std::cout << "Predicted Class: " << predicted_class << std::endl;

//     return 0;
// }



















// #include <onnxruntime_cxx_api.h>
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <algorithm>  // For std::max_element

// // Function to perform inference using ONNX Runtime
// int predict(const std::vector<float>& input_data, Ort::Session& session) {
//     // Prepare the input shape
//     std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};  // Batch size of 1

//     // Create input tensor
//     Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

//     // Get input/output names
//     const char* input_name = {"float_input"};
//     const char* output_name = {"output_label"};

//     // Run the model
//     auto output_tensors = session.Run(Ort::RunOptions{nullptr},
//                                       &input_name, &input_tensor, 1,  // Inputs
//                                       &output_name, 1);               // Outputs

//     // Get the output tensor data
//     float* output_arr = output_tensors[0].GetTensorMutableData<float>();

//     // Find the predicted class (index of the highest value in output)
//     int predicted_class = std::distance(output_arr, std::max_element(output_arr, output_arr + 4));  // Assuming 4 output classes

//     return predicted_class;  // Return the predicted class (0, 1, 2, or 3)
// }

// int main() {
//     std::vector<float> input_data(6);  // Example input size (6 features)

//     // Get input data from the user
//     std::cout << "Please enter 6 input values separated by spaces: ";
//     for (int i = 0; i < 6; ++i) {
//         std::cin >> input_data[i];
//     }

//     // Initialize ONNX Runtime environment and session
//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "classification");
//     Ort::SessionOptions session_options;
//     Ort::Session session(env, "ML_classification.onnx", session_options);  // Ensure model path is correct

//     // Perform prediction
//     int predicted_class = predict(input_data, session);
//     std::cout << "Predicted Class: " << predicted_class << std::endl;

//     return 0;
// }


#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>  // For std::max_element

// Function to perform inference using ONNX Runtime
int predict(const std::vector<float>& input_data, Ort::Session& session) {
    // Prepare the input shape (batch size = 1, feature size = 6)
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_data.size())};  // Shape = {1, 6}

    // Create input tensor (ensure that data is of type float and properly shaped)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create tensor with float type since the model expects float
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size());

    // Debug: Print the input tensor shape and size
    std::cout << "Input tensor shape: ";
    for (size_t i = 0; i < input_shape.size(); ++i) {
        std::cout << input_shape[i] << " ";
    }
    std::cout << std::endl;

    // Get input/output names (use the correct names from your model)
    const char* input_name = "input";  // Updated input name
    const char* output_name = "output_label";  // Updated output name

    // Run the model
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      &input_name, &input_tensor, 1,  // Inputs
                                      &output_name, 1);               // Output (predicted label)

    // Get the output tensor data (assuming output is of type float)
    float* output_arr = output_tensors[0].GetTensorMutableData<float>();

    // Find the predicted class (index of the highest value in output)
    int predicted_class = std::distance(output_arr, std::max_element(output_arr, output_arr + 4));  // Assuming 4 output classes

    return predicted_class;  // Return the predicted class (0, 1, 2, or 3)
}

int main() {
    std::vector<float> input_data(6);  // Input data of type float (assuming the model expects float)

    // Get input data from the user
    std::cout << "Please enter 6 input values separated by spaces: ";
    for (int i = 0; i < 6; ++i) {
        std::cin >> input_data[i];  // Input values as float
    }

    // Initialize ONNX Runtime environment and session
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "classification");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "DT.onnx", session_options);  // Ensure model path is correct

    // Perform prediction
    try {
        int predicted_class = predict(input_data, session);
        std::cout << "Predicted Class: " << predicted_class << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
    }

    return 0;
}
