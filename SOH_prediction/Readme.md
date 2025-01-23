# SOH Prediction


```bash
scp -P 2222 "/mnt/c/Users/A507658/Downloads/SoH_estimation_of_Lithium-ion_battery/4_LSTM_with_SoH/scaler_params.txt" root@localhost:/root/
scp -P 2222 "/mnt/c/Users/A507658/Downloads/SoH_estimation_of_Lithium-ion_battery/4_LSTM_with_SoH/SOH_best_model.onnx" root@localhost:/root/
scp -P 2222 -r "/mnt/c/Users/A507658/Downloads/SoH_estimation_of_Lithium-ion_battery/4_LSTM_with_SoH/soh_inference" root@localhost:/root/

```


# SoH Estimation Inference with ONNX Runtime (C++)

This project demonstrates how to perform State of Health (SoH) estimation of a Lithium-ion battery using a pre-trained machine learning model and ONNX Runtime in C++.

## Project Overview

This project does the following:

1.  **Loads a pre-trained ONNX model:** An ONNX model (e.g., `SOH_best_model.onnx`) representing a trained machine learning model is loaded. This model is expected to take battery-related features as input and output a SoH prediction.

2.  **Loads scaler parameters:** The parameters used for feature scaling during model training are loaded from a text file (e.g., `scaler_params.txt`).

3.  **Normalizes Input Data:** Input data is normalized using the loaded scaler parameters.

4.  **Performs Inference:** ONNX Runtime is used to execute inference with the loaded model and normalized input data.

5.  **Outputs Results:** The predicted SoH value, inference time, and memory usage is outputted to the console.

## Prerequisites

*   **AArch64 Linux System:** This code is primarily designed for AArch64 systems with the proper toolchain.
*   **ONNX Runtime:**  Install ONNX Runtime for C++ development (with AArch64 support)
*   **C++ Compiler:** A C++ compiler that can handle C++17.
*   **ONNX model:** You need an ONNX model of your model (e.g., `SOH_best_model.onnx`).
*   **Scaler parameters:** You need the scaler parameters of your training data to scale the data that you are using for inference.

## How to Use

1.  **Prepare ONNX model and scaler parameters:**
    *   Use the provided Python script `model_training.py` to train your model and export it to an ONNX model.
    *   Use the provided Python script `model_training.py` to save the scaling parameters in a text file.
2.  **Copy the ONNX model and scaler parameters**: Copy the `SOH_best_model.onnx` and the `scaler_params.txt` to the root directory using `scp` as explained in the documentation.
3. **Compile:**
    ```bash
    aarch64-linux-gnu-g++ -std=c++17 -o soh_inference soh_inference.cpp \
    -I/home/phani/onnxruntime-linux-aarch64-1.18.0/include \
    -L/home/phani/onnxruntime-linux-aarch64-1.18.0/lib \
    -lonnxruntime
    ```

4.  **Copy the executable:** Copy the `soh_inference` executable to the `/root` folder using scp.
5.  **Run:**
    *   Execute the `soh_inference` executable from the `/root` directory.
    ```bash
       sudo /root/soh_inference
    ```
*   This will display the predicted state of health of your battery along with the inference time and memory usage.

## Code Structure

*   **`soh_inference.cpp`:** Contains the C++ code for performing inference using ONNX Runtime. This is responsible for:
   *   Loading the ONNX model
   *   Loading the scaler parameters
   *   Normalizing input data
   *   Running Inference
   *   Printing results

* **`model_training.py`:** Python code for model training and saving the model, in ONNX format and the scaling parameters in txt format.

*   **`scaler_params.txt`:**  A text file containing the mean and standard deviation used for scaling your data.
*   **`SOH_best_model.onnx`:** The ONNX model of your ML model used for inference.

## Performance and Resource Monitoring
The C++ program also does basic performance measurements such as:
*   Inference time: Measure the amount of time that the program takes to predict the value.
*   Memory Usage: Measure the maximum resident set size (memory) in kilobytes.

## Additional Notes

*   **Error Handling:** The C++ code includes some basic error handling.
*   **Input Data:** The `test_input` array inside `soh_inference.cpp` holds the test input data. You can replace this with your own test data, and you should make sure it is consistent with the model.
*   **Optimization:** This is a basic implementation and you can optimize the code using different compiler flags and the use of hardware specific instructions.

## License

[Choose a license, or leave it as is, if you are just experimenting.]
This project is licensed under the [Your License] license.
