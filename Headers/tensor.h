#ifndef TENSOR_LIBRARY_H
#define TENSOR_LIBRARY_H
#include "stddef.h"

typedef struct {
    size_t shape[4];     // [b, c, h, w]
    size_t stride_w;     // Padded width for 64-byte alignment
    size_t total_size;   // Total elements including padding
    float* data;         // 64-byte aligned pointer
    float* grad;         // Gradient
} Tensor4D;

Tensor4D* tensor_create(size_t b, size_t c, size_t h, size_t w);                 // Create a tensor
void tensor_free(Tensor4D* t);                                                   // Deallocate memory
float tensor_get(const Tensor4D *t, size_t b, size_t c, size_t h, size_t w);     // Find a 1D value from a 4D coord
void tensor_set(Tensor4D* t, size_t b, size_t c, size_t h, size_t w, float val); // Set a 1D value from a 4D coord

// We only want to observe Tensor A and B, so we make them immutable
void tensor_matmul_2d(const Tensor4D* restrict A, const Tensor4D* restrict B, Tensor4D* restrict C); // Matrix multiplication for 2D tensors (batch and channel dimensions are ignored)
void tensor_add_bias(Tensor4D* restrict t, const float* restrict bias); // Add channel bias into every spatial pos(H * W) within that channel. We are directly modifying the tensor so it's not immutable, and we're only reading from the 1D array of bias floats
void tensor_relu(Tensor4D* restrict t); // Rectified linear unit, f(x) = (0, inf)

void tensor_transpose_OOP(const Tensor4D* restrict src, Tensor4D* restrict die); // src = source and die = derivative, transpose Out Of Place
void tensor_matmul_backwards(Tensor4D* restrict X, Tensor4D* restrict dY, Tensor4D* restrict dW, Tensor4D* restrict XT); // dW = X^T * dY
void tensor_matmul_gradient_input(Tensor4D* restrict W, Tensor4D* restrict dY, Tensor4D* restrict dX, Tensor4D* restrict WT); // dX  = dY * W^T
void tensor_accum_grad(Tensor4D* restrict target, const Tensor4D* restrict incoming_grad);

void tensor_sgd_step(Tensor4D* restrict t, float learning_rate); // Formula for Stochastic Gradient Descent: W_new = W_old - (learning_rate * gradient)
//void tensor_adam_step(Tensor4D* restrict t, );
void tensor_zero_grad(Tensor4D* restrict t);

#endif // TENSOR_LIBRARY_H
