# hpmoDNN Basic

Formatting guide
```
"C_Cpp.clang_format_fallbackStyle": "{ BasedOnStyle: Microsoft, ColumnLimit: 80, IndentWidth: 2}"
```

**GOAL:** Make a memory optimised CNN to train neural networks that don't fit in single GPU memory. 

CuDNN adn CuBLAS librarys have been used for the convoloution, pooling and fully connected primitives

**Current Progress:** Completed the framework that allows user to define a cnn and train it.

**Under Progress:** Implement a prefetch and offload policy to have memory efficiency

# Instructions 

Run the makefile to compile the code.

A basic CNN is defined in main.cu. That can be checked and run for reference and similarly one can give his own CNN for training.
