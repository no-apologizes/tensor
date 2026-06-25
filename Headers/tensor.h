#ifndef TENSOR_LIBRARY_H
#define TENSOR_LIBRARY_H
#include <stdbool.h>
#include "stddef.h"

typedef struct {
    size_t shape[4];     // [b, c, h, w]
    size_t stride_w;     // Padded width for 64-byte alignment
    size_t total_size;   // Total elements including padding
    float* data;         // 64-byte aligned pointer
    float* grad;         // Gradient
    bool is_view;
} Tensor4D;

#pragma region tensor_core.c
Tensor4D* tensor_create(size_t b, size_t c, size_t h, size_t w);                 // Create a tensor
float tensor_get(const Tensor4D *t, size_t b, size_t c, size_t h, size_t w);     // Find a 1D value from a 4D coord
void tensor_set(Tensor4D* t, size_t b, size_t c, size_t h, size_t w, float val); // Set a 1D value from a 4D coord
void tensor_free(Tensor4D* t);                                                   // Deallocate memory
void tensor_zero_data(Tensor4D* restrict t);
void tensor_zero_grad(Tensor4D* restrict t);
#pragma endregion

#pragma region tensor_io.c
void tensor_init_csv_stream(const char* restrict filename);
void tensor_reset_csv_stream(void);
int tensor_read_csv_batch(Tensor4D* restrict t, int* restrict labels, size_t batch_size);
void tensor_close_csv_stream(void);
int tensor_stream_binary(Tensor4D* restrict t, const char* restrict filename, int* restrict labels, size_t batch_size);
#pragma endregion

#pragma region tensor_ops.c
// We only want to observe Tensor A and B, so we make them immutable
void tensor_matmul_2d(const Tensor4D* restrict A, const Tensor4D* restrict B, Tensor4D* restrict C); // Matrix multiplication for 2D tensors (batch and channel dimensions are ignored)
void tensor_matmul_backwards(Tensor4D* restrict X, Tensor4D* restrict dY, Tensor4D* restrict dW, Tensor4D* restrict XT); // dW = X^T * dY
void tensor_matmul_gradient_input(Tensor4D* restrict W, Tensor4D* restrict dY, Tensor4D* restrict dX, Tensor4D* restrict WT); // dX = dY * W^T
Tensor4D* tensor_flatten_view(const Tensor4D *src);
void tensor_flatten_copy(const Tensor4D* restrict src, Tensor4D* restrict wrt);
void tensor_unflatten_copy(const Tensor4D* restrict src_grad, Tensor4D* restrict wrt_grad);
void tensor_accum_grad(Tensor4D* restrict target, const Tensor4D* restrict incoming_grad);
void tensor_add_bias(Tensor4D* restrict t, const float* restrict bias); // Add channel bias into every spatial pos(H * W) within that channel. We are directly modifying the tensor so it's not immutable, and we're only reading from the 1D array of bias floats
void tensor_transpose_OOP(const Tensor4D* restrict src, Tensor4D* restrict wrt); // src = source and wrt = write, transpose Out Of Place
float tensor_softmax_cross_entropy_loss(Tensor4D* restrict hidden, const int* restrict labels, float* restrict accuracy);
#pragma endregion

#pragma region tensor_layers.c
// 2D convolution pass where a smaller matrix slides across a 2D grid of data
// A kernel is a single 2D matrix of weights that operates on a single input channel whereas a filter operates on all channels
void tensor_conv2d(const Tensor4D* restrict input, const Tensor4D* restrict kernel, Tensor4D* restrict output);
void tensor_im2col(const Tensor4D* restrict input, size_t k_h, size_t k_w, Tensor4D* restrict output);
void tensor_col2im(const Tensor4D* restrict col, size_t b, size_t c, size_t h, size_t w, size_t k_h, size_t k_w, Tensor4D* restrict image);
void tensor_strided_conv2d_backwards(const Tensor4D* restrict dY, const Tensor4D* restrict weight, Tensor4D* restrict dX);
void tensor_strided_weight_conv2d_backwards(const Tensor4D* restrict input, const Tensor4D* restrict dY, Tensor4D* restrict dW);
void tensor_maxpool2d(const Tensor4D* restrict input, Tensor4D* restrict output, size_t* restrict indices);
void tensor_maxpool2d_backwards(const Tensor4D* restrict dY, Tensor4D* restrict dX, const size_t* restrict indices);
void tensor_layernorm(const Tensor4D* restrict src, Tensor4D* restrict wrt, const float* restrict gamma, const float* restrict beta, float epsilon);
#pragma endregion

#pragma region tensor_train.c
void tensor_relu(Tensor4D* restrict t); // Rectified linear unit, f(x) = (0, inf)
void tensor_relu_backwards(Tensor4D* restrict t);
void tensor_gelu(Tensor4D* restrict t);
void tensor_gelu_backwards(const Tensor4D* restrict pre_gelu, Tensor4D* restrict grad);
void tensor_kaiming_init(Tensor4D* restrict t, size_t fan_in); // https://docs.pytorch.org/docs/2.12/nn.init.html#:~:text=mode%20%28Literal,backwards%20pass

// Optimizers
void tensor_sgd_step(Tensor4D* restrict t, float learning_rate); // Formula for Stochastic Gradient Descent: W_new = W_old - (learning_rate * gradient)
void tensor_adam_step(Tensor4D* restrict t, Tensor4D* restrict m, Tensor4D* restrict v, // lr stands for learning rate, Adam and Muon have very different defaults
                        float lr, float beta1, float beta2, float epsilon, size_t timestep);
void tensor_muon_step(Tensor4D* restrict t, Tensor4D* restrict dW, Tensor4D* restrict X, Tensor4D* restrict XT,
                        Tensor4D* restrict Work,
                        float lr, size_t ns_steps); // Newton-Schulz steps, usually 5
#pragma endregion

#pragma region tensor_virus_hacks.c
float add_bbnos(float a, float b);
#pragma endregion

#endif // TENSOR_LIBRARY_H
