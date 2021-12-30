"# hpmoCNN"

```bash
├───.vscode
├───cnn
│   ├───include
│   ├───launch
│   └───src
└───hello_world
    ├───.vscode
    ├───launch
    └───neo
        ├───device
        └───kernels
```

```
cd cnn
nvcc -o launch\main -I ./include launch\main.cu && launch\main
```

```
cd basic
make
```

# Summary

The following layers have been implemented

1. Conv
2. Fully connected
3. Activation (Relu, sigmoid)
4. Softmax

- Using Upsampled mnist dataset.
- Working offload/prefetch mechanism with min-heap implementation.

| Batch Size   | Val_acc      | Test_acc |
| ------------ | ------------ | -------- |
| Content Cell | Content Cell | sa       |
| Content Cell | Content Cell | sa       |
