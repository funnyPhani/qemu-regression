# classification -- rf model

# classification1 -- ann model


```python
aarch64-linux-gnu-g++ -o classification1 classification_ANN.cpp     -I/home/phani/onnxruntime-linux-aarch64-1.18.0/include     -L/home/phani/onnxruntime-linux-aarch64-1.18.0/lib     -lonnxruntime
aarch64-linux-gnu-g++ -o classification classification_rf.cpp     -I/home/phani/onnxruntime-linux-aarch64-1.18.0/include     -L/home/phani/onnxruntime-linux-aarch64-1.18.0/lib     -lonnxruntime

scp -P 2222 -r /mnt/c/Users/A507658/Downloads/qwen2-docker/classification1 root@localhost:/root/
scp -P 2222 -r /mnt/c/Users/A507658/Downloads/qwen2-docker/classification root@localhost:/root/
```

![Image](https://github.com/funnyPhani/qemu-regression/blob/main/classification_task/image.png)
