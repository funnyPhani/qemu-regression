# qemu-regression


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

