# qemu-regression

```python
# convert .pkl model to onnx
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load the model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the initial types for the model inputs
initial_type = [('float_input', FloatTensorType([None, 7]))]  # 7 features

# Convert to ONNX format
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save to a file
with open('best_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

```


```bash
aarch64-linux-gnu-g++ -o model_inference model_inference.cpp -I/home/phani/onnxruntime-linux-aarch64-1.18.0/include -L/home/phani/onnxruntime-linux-aarch64-1.18.0/lib -lonnxruntime
```
```bash
scp -P 2222 -r /mnt/c/Users/A507658/Downloads/qwen2-docker/model_inference root@localhost:/root/
scp -P 2222 -r /mnt/c/Users/A507658/Downloads/qwen2-docker/best_model.onnx root@localhost:/root/
scp -P 2222 -r /mnt/c/Users/A507658/Downloads/qwen2-docker/scalar.txt root@localhost:/root/

```
```bash
# qemu
chmod +x model_inference
./model_inference
```
