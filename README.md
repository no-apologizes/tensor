A Machine Learning Engine optimized for Zen 3.

gcc (GCC) 16.1.1 20260430
cmake version 4.3.4

Full pipeline allowing for MLP and CNN neural networks.

## Features:
  Stride-row padding to 64-bytes to fit within Zen 3

  OpenMP for parallelization

  Uses linear stride arithmatic over contiguous arrays to prevent multi-dimensional pointer-chasing

  Inner matrix rows are forced to sit within CPU L1 cache
  
  - ikj instead of ijk: (tensor_matmul_2d)
    * ikj ensures that both C and B are sequentially stepping through rows and GCC can auto-vectorize this and generate AVX-512 vector instructions(Zen 3 is only AVX2(256) but 512 can be achived by double pumping)

  Out Of Place Transpose: (transpose_OOP)
 An OOP transpose keeps all data sequential and padded as transposing in place across rows and columns would destroy padded cache boundaries and be complicated to implement

  Shallow Memory Mapping: (flatten_view)
 flatten_view points to existing pre-allocated memory buffers to prevent heavy memcpy or new matrix allocation

  GEMM-Driven Spatial Convolutions: (im2col and col2im)
 Instead of using deeply nested sliding window loops im2col and col2im transforms overlapping spatial receptive patches into linear columns to prevent non-contiguous memory reading patterns inside a GEMM matrix

  Memory Resuse:
 Every Tensor4D struct is instantiated with both its data and grad pointer side-by-side so the backwards pass and easily write into them and prevents continuous scratchpad matix allocation and frequent garbage collection
 

https://app.notion.com/p/Machine-Learning-Engine-389620b9962b80c486f0edee8021b5ad
