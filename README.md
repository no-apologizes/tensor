A Machine Learning Engine/Library optimized for Zen 3[^1] cache lines.    
Full pipeline allowing for MLP and CNN neural networks.    
Counting *.h and *c files, there are 817 lines of code

AI was used for pseudo code generation, research, and light debugging[^2].    
Research verified with sources such as academic papers and unclear topics were decided upon the most agreed method.

gcc (GCC) 16.1.1 20260430    
cmake version 4.3.4

## Features:
  Stride-row padding to 64-bytes to fit within Zen 3

  OpenMP for parallelization

  Uses linear stride arithmatic over contiguous arrays to prevent multi-dimensional pointer-chasing

  Inner matrix rows are forced to sit within CPU L1 cache
  
  - ikj instead of ijk: (tensor_matmul_2d)
    * ikj ensures that both C and B are sequentially stepping through rows and GCC can auto-vectorize and generate AVX-512 vector instructions(Zen 3 is only AVX2(256) but 512 can be achived by double pumping)

  - Out Of Place Transpose: (transpose_OOP)
    * An OOP transpose keeps all data sequential and padded as transposing in place across rows and columns would destroy padded cache boundaries and be complicated to implement

  - Shallow Memory Mapping: (flatten_view)
    * flatten_view points to existing pre-allocated memory buffers to prevent heavy memcpy or new matrix allocation

  - GEMM-Driven Spatial Convolutions: (im2col and col2im)
    * Instead of using deeply nested sliding window loops im2col and col2im transforms overlapping spatial receptive patches into linear columns to prevent non-contiguous memory reading patterns inside a GEMM matrix

  - Memory Resuse:
    * Every Tensor4D struct is instantiated with both its data and gradient pointer side-by-side so the backwards pass and easily write into them and prevents continuous scratchpad matix allocation and frequent garbage collection

Basic pipeline of neural network:    
https://app.notion.com/p/Machine-Learning-Engine-389620b9962b80c486f0edee8021b5ad

unsigned_char_subtract within [tensor_virus_hacks.c](../master/tensor_virus_hacks) is a joke function and shouldn't be here but I can do whatever I want.

[^1]: Changing -march=znver3 to -march=znver4 in [CMakeLists.txt](../master/CMakeLists.txt) works and GCC may vectorize SIMD differently
[^2]: Light debugging as fixing loop order or accidentally writing over data instead of gradients
