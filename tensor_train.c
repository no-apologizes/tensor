#include <math.h>
#include <stdlib.h>

#include "Headers/tensor.h"

void tensor_relu(Tensor4D* restrict t) { // f(x) = (0, inf), that's fancy speak for x < 0 = 0 and x > = 0 = x
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
        // We also don't care to skip across the data using the strided width or whatever because it's garbage memory and all the actual functions use strided_w
    }
}

void tensor_relu_backwards(Tensor4D* restrict t) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        if (t->data[i] <= 0.0f) {
            t->grad[i] = 0.0f; // Zero grad for backward pass
        }
    }
}

void tensor_gelu(Tensor4D* restrict t) {
    const float kNormal = 0.797884561f;
    const float kGeluCoef = 0.044715f;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        const float a = t->data[i];

        const float a_cube = a * a * a;
        const float inner = kNormal * (a + kGeluCoef * a_cube);
        t->data[i] = 0.5f * a * (1.0f + tanhf(inner));
    }
}

void tensor_gelu_backwards(const Tensor4D* restrict pre_gelu, Tensor4D* restrict grad) {
    // pre_gelu is the original matrix output BEFORE it was modified by the forward gelu
    // grad is the tensor containing the incoming backprop gradients

    const float k = 0.7978845608f; // sqrt(2 / pi)
    const float c = 0.044715f;     // gelu cubic coefficient0

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < pre_gelu->total_size; i++) {
        const float x = pre_gelu->data[i];
        const float x2 = x * x;
        const float x3 = x2 * x;

        // Hyperbolic tangent core
        const float inner = k * (x + c * x3);
        const float tanh_inner = tanhf(inner);

        // Calculate the two halves of the derivative
        const float left_term = 0.5f * (1.0f + tanh_inner);
        const float right_term = 0.5f * x * (1.0f - tanh_inner * tanh_inner) * k * (1.0f + 3.0f * c * x2);

        // Chain Rule
        grad->grad[i] *= (left_term + right_term);
    }
}

static inline float rand_norm() { // Generate a normal distribution with the Box-Muller transform; mean=0, standard deviation=1
    // Avoid log(0) [undefined]
    const float u1 = ((float)rand() / (float)RAND_MAX) + 1e-7f;
    const float u2 = (float)rand() / (float)RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 6.283185307179586f * u2); // 6.283185307179586 ~ tau
}

void tensor_kaiming_init(Tensor4D* restrict t, size_t fan_in) { // Just look it up dude
    // https://app.notion.com/p/Something-372620b9962b8005bdb1d7eeccfb3b4c
    const float stddev = sqrtf(2.0f / (float)fan_in);

    const size_t batches  = t->shape[0];
    const size_t channels = t->shape[1];
    const size_t height   = t->shape[2];
    const size_t width    = t->shape[3];

    //#pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                size_t row_offset = (b * channels * height * t->stride_w) +
                                    (c * height * t->stride_w) +
                                    (h * t->stride_w);

                for (size_t w = 0; w < width; w++) {
                    // Init active data while skipping over padding boundaries
                    t->data[row_offset + w] = rand_norm() * stddev;
                }
            }
        }
    }
}

void tensor_sgd_step(Tensor4D* restrict t, float learning_rate) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        // Formula for SGD
        // W_new = W_old - (learning_rate * gradient)
        t->data[i] -= learning_rate * t->grad[i];
    }
}

void tensor_adam_step(Tensor4D* restrict t, Tensor4D* restrict m, Tensor4D* restrict v, float lr, float beta1, float beta2, float epsilon, size_t timestep) {
    const float bias_cor1 = 1.0f - powf(beta1, (float)timestep);
    const float bias_cor2 = 1.0f - powf(beta2, (float)timestep);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        const float grad = t->grad[i];

        // Update based on first moment
        m->data[i] = (beta1 * m->data[i]) + ((1.0f - beta1) * grad);

        // Update based on second moment
        v->data[i] = (beta2 * v->data[i]) + ((1.0f - beta2) * grad);

        // Compute bias-corrected versions(hat as in m^ and v^)
        const float m_hat = m->data[i] / bias_cor1;
        const float v_hat = v->data[i] / bias_cor2;

        t->data[i] -= (lr / (sqrtf(v_hat) + epsilon)) * m_hat;
    }
}

void tensor_muon_step(Tensor4D* restrict t, Tensor4D* restrict dW, Tensor4D* restrict X, Tensor4D* restrict XT, Tensor4D* restrict Work, float lr, size_t ns_steps) {
    // Muon operates on flattened 2D dimensions of the weight matrix
    //const size_t N = dW->shape[2]; N is unused because of the formula
    const size_t M = dW->shape[3];

    // Normalize the grad matrix by its supergalactic-orthodivergent-extralinear-hyperisomorphic scalar
    float frobenius_sum = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:frobenius_sum)
    for (size_t i = 0; i < dW->total_size; i++) {
        frobenius_sum += dW->data[i] * dW->data[i];
    }
    const float inv_norm = 1.0f / (sqrtf(frobenius_sum) + 1e-7f); // Inverse normal

    // Copy normalized dW into working matrix X
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dW->total_size; i++) {
        X->data[i] = dW->data[i] * inv_norm;
    }

    // Newton-Schulz orthogonalization loop
    // X = 0.5 * X * (3I - XT * X) [Most compact factored form]
    for (size_t step = 0; step < ns_steps; step++) {
        // Transpose current X into XT
        tensor_transpose_OOP(X, XT);

        // Compute Work = XT * X (Sized M x M)
        tensor_zero_data(Work);
        tensor_matmul_2d(XT, X, Work);

        // Work = 3I - Work
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t remote = 0; remote < M; remote++) {
            for (size_t andie = 0; andie < M; andie++) {
                size_t index = (remote * Work->stride_w) + andie;
                // Generate a scaled Identity matrix, 3.0 along the main diagonal and zeros everywhere else
                float identity = (remote == andie) ? 3.0f : 0.0f;
                Work->data[index] = identity - Work->data[index];
            }
        }

        // Reuse XT buffer to hold intermediate result
        tensor_zero_data(XT);
        tensor_matmul_2d(X, Work, XT); // XT = X * Work

        // Update X for the next iteration
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < X->total_size; i++) {
            X->data[i] = 0.5f * XT->data[i];
        }
    }

    // Apply orthogonal update to weights
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] -= lr * X->data[i];
    }
}