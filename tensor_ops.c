#include <math.h>
#include <stdlib.h>

#include "Headers/tensor.h"
void tensor_matmul_2d(const Tensor4D* restrict A, const Tensor4D* restrict B, Tensor4D* restrict C) {
    const size_t batches  = A->shape[0];
    const size_t channels = A->shape[1];

    const size_t M        = A->shape[2]; // Rows of A
    const size_t K        = A->shape[3]; // Inner dimension (Cols of A / Rows of B)
    const size_t N        = B->shape[3]; // Cols of B

    const int broadcast_C = (C->shape[0] == 1);

    const size_t b_stride_B = (B->shape[0] == 1) ? 0 : 1;
    const size_t b_stride_C = (C->shape[0] == 1) ? 0 : 1;
    const size_t c_stride_B = (B->shape[1] == 1) ? 0 : 1;
    const size_t c_stride_A = (A->shape[1] == 1) ? 0 : 1;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            // Isolate the exact 2D memory planes using the padded stride tracking parameters
            const size_t offset_A = (b * A->shape[1] * A->shape[2] * A->stride_w) + ((c * c_stride_A) * A->shape[2] * A->stride_w);
            const size_t offset_B = ((b * b_stride_B) * B->shape[1] * B->shape[2] * B->stride_w) + ((c * c_stride_B) * B->shape[2] * B->stride_w);
            const size_t offset_C = ((b * b_stride_C) * C->shape[1] * C->shape[2] * C->stride_w) + (c * C->shape[2] * C->stride_w);

            const float* restrict slice_A = &A->data[offset_A];
            const float* restrict slice_B = &B->data[offset_B];
            float* restrict slice_C       = &C->data[offset_C];

            for (size_t i = 0; i < M; i++) {
                const float* restrict row_A = &slice_A[i * A->stride_w];
                float* restrict row_C       = &slice_C[i * C->stride_w];

                for (size_t k = 0; k < K; k++) {
                    const float val_A = row_A[k];
                    const float* restrict row_B = &slice_B[k * B->stride_w];

                    if (broadcast_C) {
                        for (size_t j = 0; j < N; j++) {
                            #pragma omp atomic
                            row_C[j] += val_A * row_B[j];
                        }
                    } else {
                        #pragma clang loop vectorize(enable) interleave(enable)
                        for (size_t j = 0; j < N; j++) {
                            row_C[j] += val_A * row_B[j];
                        }}}}}}
    #pragma omp barrier
}

void tensor_matmul_backwards(Tensor4D* restrict X, Tensor4D* restrict dY, Tensor4D* restrict dW, Tensor4D* restrict XT) {
    // OOP Transpose
    tensor_transpose_OOP(X, XT);

    // Compute weight gradient
    // dW = XT * dY
    tensor_matmul_2d(XT, dY, dW);
    // Accumulate directly into the weight gradient buffer
    // Make sure to zero out dW->data
    // With tensor_zero_grad
    // But only ONCE at the start of every batch
}

void tensor_matmul_gradient_input(Tensor4D* restrict W, Tensor4D* restrict dY, Tensor4D* restrict dX, Tensor4D* restrict WT) {
    // Transpose W into WT
    tensor_transpose_OOP(W, WT);

    // dX  = dY * W^T
    tensor_matmul_2d(dY, WT, dX);
    // Make sure to zero out dX->data with tensor_zero_grad
    // But only ONCE at the start of every batch
}

Tensor4D* tensor_flatten_view(const Tensor4D* src) {
    if (!src) return NULL;

    // Allocate a tiny amount of memory for new struct shell
    Tensor4D* view = malloc(sizeof(Tensor4D));
    if (!view) return NULL;

    // Inherit
    view->data = src->data;
    view->grad = src->grad;

    // Flatten axes 1, 2, and 3 down into width
    view->shape[0] = src->shape[0]; // Batch size remains identical
    view->shape[1] = 1;
    view->shape[2] = 1;
    view->shape[3] = src->shape[1] * src->shape[2] * src->shape[3]; // 1 * 28 * 28 = 784

    // Inherit the padded stride alignment
    view->stride_w  = src->stride_w;
    view->total_size = view->shape[0] * view->shape[1] * view->shape[2] * view->stride_w;

    // Declare that this struct does not own its memory buffers
    view->is_view = true;

    return view;
}

void tensor_flatten_copy(const Tensor4D* restrict src, Tensor4D* restrict wrt) {
    const size_t batches  = src->shape[0];
    const size_t channels = src->shape[1];
    const size_t height   = src->shape[2];
    const size_t width    = src->shape[3];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        float* restrict wrt_row = &wrt->data[b * wrt->stride_w];

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                const size_t src_offset = (b * channels * height * src->stride_w) +
                                            (c * height * src->stride_w) +
                                                (h * src->stride_w);

                const size_t wrt_base = (c * height * width) + (h * width);

                #pragma omp simd
                for (size_t w = 0; w < width; w++) {
                    wrt_row[wrt_base + w] = src->data[src_offset + w];
                }
            }
        }
    }
}

void tensor_unflatten_copy(const Tensor4D* restrict src_grad, Tensor4D* restrict wrt_grad) {
    const size_t batches  = wrt_grad->shape[0];
    const size_t channels = wrt_grad->shape[1];
    const size_t height   = wrt_grad->shape[2];
    const size_t width    = wrt_grad->shape[3];

    #pragma omp for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        const float* restrict src_row = &src_grad->grad[b * src_grad->stride_w];

        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                const size_t dst_offset = (b * channels * height * wrt_grad->stride_w) +
                                          (c * height * wrt_grad->stride_w) +
                                          (h * wrt_grad->stride_w);

                const size_t src_base = (c * height * width) + (h * width);

                #pragma omp simd
                for (size_t w = 0; w < width; w++) {
                    wrt_grad->grad[dst_offset + w] += src_row[src_base + w];
                }
            }
        }
    }
}

void tensor_accum_grad(Tensor4D* restrict target, const Tensor4D* restrict incoming_grad) {
    // Bound the loop strictly to the structural allocation limits of the destination target tensor
    const size_t loop_bounds = target->total_size;

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < loop_bounds; i++) {
        target->grad[i] += incoming_grad->data[i];
    }
}

void tensor_add_bias(Tensor4D* restrict t, const float* restrict bias) {
    // BCHW, unpack the 4D coordinate shapes into local vars to prevent the cpu from dereferencing the t struct pointer every iteration
    const size_t batches  = t->shape[0];
    const size_t channels = t->shape[1];
    const size_t height   = t->shape[2];
    const size_t width    = t->shape[3];

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            // Because a bias vector applies a single scalar value across an entire plane, we look it up exactly once right here so the cpu never has to look up the bias array pointer again
            const float b_val = bias[c];
            for (size_t h = 0; h < height; h++) { // Step vertically through the current 2D matrix slice
                // Get pointer to the start of this specific row
                float* restrict row_ptr = &t->data[(b * channels * height * t->stride_w) + (c * height * t->stride_w) + (h * t->stride_w)];
#pragma clang loop vectorize(enable)
                for (size_t w = 0; w < width; w++) { // Here we step horizontally along the columns of our row
                    row_ptr[w] += b_val;
                }}}}
}

void tensor_transpose_OOP(const Tensor4D* restrict src, Tensor4D* restrict wrt) {
    const size_t tile_size = 32; // cpu cache is 32x32

    const size_t batches  = src->shape[0];
    const size_t channels = src->shape[1];
    const size_t src_h    = src->shape[2];
    const size_t src_w    = src->shape[3];

    // Create threads and spread execution across them
    // Merge inner and outer loop into a single large iteration space
    // Spreads the iterations across created threads evenly
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            const size_t offset_src = (b * channels * src_h * src->stride_w) + (c * src_h * src->stride_w);
            const size_t offset_wrt = (b * channels * src_w * wrt->stride_w) + (c * src_w * wrt->stride_w);

            const float* restrict slice_src = &src->data[offset_src];
            float* restrict slice_wrt = &wrt->data[offset_wrt];

            for (size_t i = 0; i < src_h; i += tile_size) {
                for (size_t j = 0; j < src_w; j += tile_size) {
                    for (size_t remote = 0; remote < tile_size && (j + remote < src_w); remote++) {
                        for (size_t andie = 0; andie < tile_size && (i + andie < src_h); andie++) {

                            // Source index: Row major layout (Row * stride + Col)
                            const size_t src_idx = (i + andie) * src->stride_w + (j + remote);

                            // Write index: Transposed coordinates (Col becomes Row, Row becomes Col)
                            const size_t wrt_idx = (j + remote) * wrt->stride_w + (i + andie);

                            slice_wrt[wrt_idx] = slice_src[src_idx];
                        }}}}}}
}

float tensor_softmax_cross_entropy_loss(Tensor4D* restrict hidden, const int* restrict labels, float* restrict accuracy) {
    const size_t batches = hidden->shape[0];
    const size_t classes = hidden->shape[3]; // hidden_dim = 10

    float total_loss = 0.0f;
    size_t correct_count = 0;


    for (size_t b = 0; b < batches; b++) {
        const float* restrict row_data = &hidden->data[b * hidden->stride_w];
        float* restrict row_grad = &hidden->grad[b * hidden->stride_w];
        const int target = labels[b];

        // Find largest number and subtract it from everything to prevent overflow
        float max_val = row_data[0];
        for (size_t j = 1; j < classes; j++) {
            if (row_data[j] > max_val) max_val = row_data[j];
        }

        // Force all negative logits to become positive and sum_exp acts as the 100%
        float sum_exp = 0.0f;
        for (size_t j = 0; j < classes; j++) {
            row_grad[j] = expf(row_data[j] - max_val); // Temp store exp
            sum_exp += row_grad[j];
        }

        // Compute Softmax Probabilities
        for (size_t j = 0; j < classes; j++) {
            row_grad[j] /= sum_exp; // row_grad now contains true softmax probabilities
        }

        // Loss calculation, femtofarad prevents the log of 0
        total_loss -= logf(row_grad[target] + 1e-15f);

        size_t max_idx = 0;
        float max_prob = row_grad[0];
        for (size_t j = 1; j < classes; j++) {
            if (row_grad[j] > max_prob) {
                max_prob = row_grad[j];
                max_idx = j;
            }
        }
        if (max_idx == (size_t)target) {
            correct_count++;
        }

        // Compute Cross-Entropy Gradient: (softmax_probability - target_one_hot) / batch_size
        for (size_t j = 0; j < classes; j++) {
            if (j == (size_t)target) {
                row_grad[j] = (row_grad[j] - 1.0f) / (float)batches;
            } else {
                row_grad[j] = row_grad[j] / (float)batches;
            }
        }
    }

    *accuracy = (float)correct_count / (float)batches;
    return total_loss / (float)batches;
}