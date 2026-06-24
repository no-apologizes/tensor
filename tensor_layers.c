#include "Headers/tensor.h"
#include <math.h>

void tensor_conv2d(const Tensor4D* restrict input, const Tensor4D* restrict kernel, Tensor4D* restrict output) {
    const size_t batches = input->shape[0]; // Batches
    const size_t in_ch   = input->shape[1]; // Input channels
    const size_t in_h    = input->shape[2];
    const size_t in_w    = input->shape[3];

    const size_t out_ch  = kernel->shape[0]; // Output channels
    const size_t k_h     = kernel->shape[2]; // Kernel height and width
    const size_t k_w     = kernel->shape[3];

    const size_t out_h   = in_h - k_h + 1; // Valid padding spatial output formula
    const size_t out_w   = in_w - k_w + 1;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t oc = 0; oc < out_ch; oc++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                size_t out_row_offset = (b * out_ch * out_h * output->stride_w) +
                                        (oc * out_h * output->stride_w) +
                                        (oh * output->stride_w);

                for (size_t ow = 0; ow < out_w; ow++) {
                    float out_val = 0.0f;

                    for (size_t ic = 0; ic < in_ch; ic++) {
                        for (size_t kh = 0; kh < k_h; kh++) {
                            size_t in_row_offset = (b * in_ch * in_h * input->stride_w) +
                                                   (ic * in_h * input->stride_w) +
                                                   ((oh + kh) * input->stride_w);

                            size_t ker_row_offset = (oc * in_ch * k_h * kernel->stride_w) +
                                                    (ic * k_h * kernel->stride_w) +
                                                    (kh * kernel->stride_w);

                            #pragma omp simd reduction(+:out_val)
                            for (size_t kw = 0; kw < k_w; kw++) {
                                out_val += input->data[in_row_offset + ow + kw] * kernel->data[ker_row_offset + kw];
                            }}}
                    output->data[out_row_offset + ow] = out_val;
                }}}}
}

void tensor_im2col(const Tensor4D* restrict input, size_t k_h, size_t k_w, Tensor4D* restrict output) {
    const size_t batches = input->shape[0];
    const size_t in_ch   = input->shape[1];
    const size_t in_h    = input->shape[2];
    const size_t in_w    = input->shape[3];

    const size_t out_h   = in_h - k_h + 1;
    const size_t out_w   = in_w - k_w + 1;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                // Calculate which sequential row this patch maps to in the destination GEMM matrix
                size_t col_row = (b * out_h * out_w) + (oh * out_w) + ow;
                size_t out_row_offset = col_row * output->stride_w;

                size_t col_idx = 0;
                for (size_t ic = 0; ic < in_ch; ic++) {
                    for (size_t kh = 0; kh < k_h; kh++) {
                        size_t in_row_offset = (b * in_ch * in_h * input->stride_w) +
                                               (ic * in_h * input->stride_w) +
                                               ((oh + kh) * input->stride_w);

                        // Contiguous vector width copy
                        #pragma omp simd
                        for (size_t kw = 0; kw < k_w; kw++) {
                            output->data[out_row_offset + col_idx] = input->data[in_row_offset + ow + kw];
                            col_idx++;
                        }}}}}}
}

void tensor_col2im(const Tensor4D* restrict col, size_t b, size_t c, size_t h, size_t w, size_t k_h, size_t k_w, Tensor4D* restrict image) {
    const size_t out_h = h - k_h + 1;
    const size_t out_w = w - k_w + 1;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t batch_idx = 0; batch_idx < b; batch_idx++) {
        for (size_t oh = 0; oh < out_h; oh++) {
            for (size_t ow = 0; ow < out_w; ow++) {
                size_t patch_row = (batch_idx * out_h * out_w) + (oh * out_w) + ow;
                size_t col_row_offset = patch_row * col->stride_w;

                size_t col_idx = 0;
                for (size_t ic = 0; ic < c; ic++) {
                    for (size_t kh = 0; kh < k_h; kh++) {
                        size_t img_row_offset = (batch_idx * c * h * image->stride_w) +
                                               (ic * h * image->stride_w) +
                                               ((oh + kh) * image->stride_w);

                        // Because sliding windows overlap spatially, threads must accumulate gradients atomically
                        for (size_t kw = 0; kw < k_w; kw++) {
                            #pragma omp atomic
                            image->grad[img_row_offset + ow + kw] += col->data[col_row_offset + col_idx];
                            col_idx++;
                        }}}}}}
}

void tensor_strided_conv2d_backwards(const Tensor4D* restrict dY, const Tensor4D* restrict weight, Tensor4D* restrict dX) {
    const size_t batches = dX->shape[0];
    const size_t in_ch   = dX->shape[1];
    const size_t in_h    = dX->shape[2];
    // in_w not needed

    const size_t out_ch  = weight->shape[0];
    const size_t k_h     = weight->shape[2];
    const size_t k_w     = weight->shape[3];

    const size_t out_h   = dY->shape[2];
    const size_t out_w   = dY->shape[3];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            for (size_t oc = 0; oc < out_ch; oc++) {
                for (size_t oh = 0; oh < out_h; oh++) {
                    const size_t dY_row_offset = (b * out_ch * out_h * dY->stride_w) +
                                            (oc * out_h * dY->stride_w) +
                                                (oh * dY->stride_w);

                    for (size_t kh = 0; kh < k_h; kh++) {
                        const size_t dX_row_offset = (b * in_ch * in_h * dX->stride_w) +
                                                (ic * in_h * dX->stride_w) +
                                                ((oh + kh) * dX->stride_w);

                        const size_t ker_row_offset = (oc * in_ch * k_h * weight->stride_w) +
                                                    (ic * k_h * weight->stride_w) +
                                                        (kh * weight->stride_w);

                        for (size_t kw = 0; kw < k_w; kw++) {
                            const float w_val = weight->data[ker_row_offset + kw];

                            // Direct vector assignment, no atomic locks
                            #pragma omp simd
                            for (size_t ow = 0; ow < out_w; ow++) {
                                dX->grad[dX_row_offset + ow + kw] += dY->grad[dY_row_offset + ow] * w_val;
                            }}}}}}}
}

void tensor_strided_weight_conv2d_backwards(const Tensor4D* restrict input, const Tensor4D* restrict dY, Tensor4D* restrict dW) {
    const size_t batches = input->shape[0];
    const size_t in_ch   = input->shape[1];
    const size_t in_h    = input->shape[2];
    const size_t in_w    = input->shape[3];

    const size_t out_ch  = dW->shape[0];
    const size_t k_h     = dW->shape[2];
    const size_t k_w     = dW->shape[3];

    const size_t out_h   = dY->shape[2];
    const size_t out_w   = dY->shape[3];

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t oc = 0; oc < out_ch; oc++) {
        for (size_t ic = 0; ic < in_ch; ic++) {
            for (size_t kh = 0; kh < k_h; kh++) {
                size_t dW_row_offset = (oc * in_ch * k_h * dW->stride_w) +
                                       (ic * k_h * dW->stride_w) +
                                       (kh * dW->stride_w);

                for (size_t kw = 0; kw < k_w; kw++) {
                    float grad_val = 0.0f;

                    for (size_t b = 0; b < batches; b++) {
                        for (size_t oh = 0; oh < out_h; oh++) {
                            size_t dY_row_offset = (b * out_ch * out_h * dY->stride_w) +
                                                   (oc * out_h * dY->stride_w) +
                                                   (oh * dY->stride_w);

                            size_t in_row_offset = (b * in_ch * in_h * input->stride_w) +
                                                   (ic * in_h * input->stride_w) +
                                                   ((oh + kh) * input->stride_w);

                            // Innermost loop moves horizontally across width
                            #pragma omp simd reduction(+:grad_val)
                            for (size_t ow = 0; ow < out_w; ow++) {
                                grad_val += dY->grad[dY_row_offset + ow] * input->data[in_row_offset + ow + kw];
                            }
                        }
                    }
                    // Accumulate safely into the pre-allocated dW gradient buffer
                    dW->grad[dW_row_offset + kw] += grad_val;
                }}}}
}

void tensor_maxpool2d(const Tensor4D* restrict input, Tensor4D* restrict output, size_t* restrict indices) {
    const size_t batches  = input->shape[0];
    const size_t channels = input->shape[1];
    const size_t in_h     = input->shape[2];
    // in_w isn't used
    const size_t out_h    = output->shape[2];
    const size_t out_w    = output->shape[3];

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                const size_t ih_start = oh * 2;
                const size_t out_offset = (b * channels * out_h * output->stride_w) +
                                        (c * out_h * output->stride_w) +
                                            (oh * output->stride_w);

                const size_t in_base_offset = (b * channels * in_h * input->stride_w) +
                                            (c * in_h * input->stride_w);

                for (size_t ow = 0; ow < out_w; ow++) {
                    const size_t iw_start = ow * 2;

                    // Parts of a 2x2 kernel
                    const size_t idx00 = in_base_offset + (ih_start * input->stride_w) + iw_start; // Top left
                    const size_t idx01 = idx00 + 1;                                                // Top right
                    const size_t idx10 = idx00 + input->stride_w;                                  // Bottom left
                    const size_t idx11 = idx10 + 1;                                                // Bottom right

                    // Values
                    const float val00 = input->data[idx00];
                    const float val01 = input->data[idx01];
                    const float val10 = input->data[idx10];
                    const float val11 = input->data[idx11];

                    float max_val = val00;
                    size_t max_idx = idx00;

                    // Fina highest
                    if (val01 > max_val) { max_val = val01; max_idx = idx01; }
                    if (val10 > max_val) { max_val = val10; max_idx = idx10; }
                    if (val11 > max_val) { max_val = val11; max_idx = idx11; }

                    output->data[out_offset + ow] = max_val;

                    const size_t out_flat_idx = out_offset + ow;
                    indices[out_flat_idx] = max_idx;
                }}}}
}

void tensor_maxpool2d_backwards(const Tensor4D* restrict dY, Tensor4D* restrict dX, const size_t* restrict indices) {
    const size_t batches  = dY->shape[0];
    const size_t channels = dY->shape[1];
    const size_t out_h    = dY->shape[2];
    const size_t out_w    = dY->shape[3];

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t oh = 0; oh < out_h; oh++) {
                const size_t out_offset = (b * channels * out_h * dY->stride_w) +
                                        (c * out_h * dY->stride_w) +
                                            (oh * dY->stride_w);

                for (size_t ow = 0; ow < out_w; ow++) {
                    const size_t out_flat_idx = out_offset + ow;
                    const size_t target_in_idx = indices[out_flat_idx];

                    #pragma omp atomic
                    dX->grad[target_in_idx] += dY->grad[out_flat_idx];
                }}}}
}

void tensor_layernorm(const Tensor4D *src, Tensor4D *wrt, const float *gamma, const float *beta, float epsilon) {
    const size_t batches  = src->shape[0];
    const size_t channels = src->shape[1];
    const size_t height   = src->shape[2];
    const size_t width    = src->shape[3];

    // Calculate active pixels
    const size_t elements_per_slice = channels * height * width;
    const size_t inv_elements = 1.0f / (float)elements_per_slice; // Inverse standard deviation

    // Total physical rows in a single batch slice
    const size_t rows_per_slice = channels * height;

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < batches; b++) {
        // Starting index
        const size_t batch_offset = b * rows_per_slice * src->stride_w;

        // Compute mean
        float sum = 0.0f;

        for (size_t row = 0; row < rows_per_slice; row++) {
            const float* restrict row_ptr = &src->data[batch_offset + (row * src->stride_w)];

            // Split sum across vector registers
            #pragma omp simd reduction(+:sum)
            for (size_t w = 0; w < width; w++) {
                sum += row_ptr[w];
            }
        }
        const float mean = sum * inv_elements;

        // Compute variance
        float variance_sum = 0.0f;

        for (size_t row = 0; row < rows_per_slice; row++) {
            const float* restrict row_ptr = &src->data[batch_offset + (row * src->stride_w)];

            #pragma omp simd reduction(+:variance_sum)
            for (size_t w = 0; w < width; w++) {
                const float diff = row_ptr[w] - mean;
                variance_sum += diff * diff;
            }
        }
        const float variance = variance_sum * inv_elements;
        const float inv_stddev = 1.0f / sqrtf(variance + epsilon); // Inverse standard deviation

        // Un collapse C and H because Gamma and Beta are channel specific
        for (size_t c = 0; c < channels; c++) {
            const float g_val = gamma[c];
            const float b_val = beta[c];

            for (size_t h = 0; h < height; h++) {
                // Calculate flat row index manually
                const size_t row_idx = (c * height) + h;

                const float* restrict in_ptr = &src->data[batch_offset + (row_idx * src->stride_w)];
                float* restrict out_ptr = &wrt->data[batch_offset + (row_idx * wrt->stride_w)];

                #pragma omp simd
                for (size_t w = 0; w < width; w++) {
                    const float norm_val = (in_ptr[w] - mean) * inv_stddev;
                    out_ptr[w] = (norm_val * g_val) + b_val;
                }}}}
}