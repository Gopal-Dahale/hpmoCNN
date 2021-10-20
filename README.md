"# hpmoCNN" 

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
cd cnn
nvcc -o launch\main -I ./include launch\main.cu && launch\main
```