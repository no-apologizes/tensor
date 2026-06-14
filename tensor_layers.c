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

                            #pragma omp simd
                            for (size_t kw = 0; kw < k_w; kw++) {
                                out_val += input->data[in_row_offset + ow + kw] * kernel->data[ker_row_offset + kw];
                            }}}
                    output->data[out_row_offset + ow] = out_val;
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