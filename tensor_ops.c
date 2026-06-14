#include "Headers/tensor.h"

void tensor_matmul_2d(const Tensor4D* restrict A, const Tensor4D* restrict B, Tensor4D* restrict C) { // We can use restrict here because matrix A and B are completely separate things
    // We have batches and channels because we need to flatten a 4D matrix into a 2D matrix
    const size_t batches  = A->shape[0];
    const size_t channels = A->shape[1];
    /*
    Matrix A has N rows and K columns
    A = N * K

    Matrix B has K rows and M columns
    B = K * M

    C = N * M
    The inner dimension K must match for the multiplication to be valid,
    and the resulting matrix C will have the outer dimensions of A and B (N from A and M from B)
    */
    const size_t M        = A->shape[2];
    const size_t K        = A->shape[3];
    const size_t N        = B->shape[3];

    // See if we're writing to shared weight matrix (gradient accumulation)
    const int broadcast_C = (C->shape[0] == 1);

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) { // Anyway we can take advantage of the fact that we know the limits of the channels? Like unsigned 8-bits bc of the 0-255 color range | No, size_t already matches the native pointer size of the cpu registers and if we force it to use smaller thing, it has to spend extra clock cycles on zero or sign-extending to fit the 64-bit register
        for (size_t c = 0; c < channels; c++) { // Also I'm aware of size_t being the standard for tracking incrementation in loops in such for C, but I'm on linux and I don't care to make it work on Windows so is there anything I can do to make it 'faster'? Or even take less memory. These are useless things to change, but I'm curious | Yes, we can use pointer aliasing to allow the compiler to preform even more aggressive optimizations
            // Isolate the exact 2D memory planes using the padded stride
            const size_t offset_A = (b * A->shape[1] * A->shape[2] * A->stride_w) + (c * A->shape[2] * A->stride_w);
            const size_t offset_B = (b * B->shape[1] * B->shape[2] * B->stride_w) + (c * B->shape[2] * B->stride_w);
            const size_t offset_C = (b * C->shape[1] * C->shape[2] * C->stride_w) + (c * C->shape[2] * C->stride_w);

            // Use a direct pointer to the first element of the 2D plane
            const float* slice_A = &A->data[offset_A];
            const float* slice_B = &B->data[offset_B];
            float* slice_C = &C->data[offset_C];

            // Cache blocking constants to match zen 3 L1 and L2 sizes
            // Size of 64 allows for 64 floats (256 bytes)
            const size_t BLOCK_I = 64;
            const size_t BLOCK_K = 64;

            for (size_t i0 = 0; i0 < M; i0 += BLOCK_I) {
                const size_t i_max = (i0 + BLOCK_I < M) ? i0 + BLOCK_I : M;
                for (size_t k0 = 0; k0 < K; k0 += BLOCK_K) {
                    const size_t k_max = (k0 + BLOCK_K < K) ?  k0 + BLOCK_K : K;

                    for (size_t i = i0; i < i_max; i++) {
                        const size_t i_stride_A = i * A->stride_w;
                        const size_t i_stride_C = i * C->stride_w;

                        for (size_t k = k0; k < k_max; k++) {
                            const float val_A = slice_A[i_stride_A + k];
                            const size_t k_stride_B = k * B->stride_w;

                            if (broadcast_C) {
                                for (size_t j = 0;  j < N; j++) {
                                    const float update = val_A * slice_B[k_stride_B + j];
                                    #pragma omp atomic
                                    slice_C[i_stride_C + j] += update;
                                }
                            } else {
                                #pragma clang loop vectorize(enable) interleave(enable)
                                for (size_t j = 0; j < N; j++) {
                                    slice_C[i_stride_C + j] += val_A * slice_B[k_stride_B + j];
                            }}}}}}}}
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

                    for (size_t remote = j; remote < j + tile_size && remote < src_w; remote++) {
                        // Pre-calculate the row offset for the derivative matrix (die)
                        const size_t s_stride_src = remote * src->stride_w;
                        for (size_t andie = i; andie < i + tile_size && andie < src_h; andie++) {
                            // In a transpose operation, either the src read or the dst write must be strided
                            // In here we have src strided because it's faster to read data because the prefetcher thinks we want to read it
                            // Sequential writes and strided reads are a much better tradeoff then sequential reads and strided writes because it's much slower as the
                            // cpu has to jump by the strided width which causes a whole bunch of other problems
                            // That's only for older hardware though and Zen 3 and modern cpus have a thing called store buffers so we get that and sequential reads
                            // So sequential reads and strided writes are faster
                            // Nvm it's the other way around
                            // So, wait I already had that, never mind idk anymore
                            slice_wrt[andie * wrt->stride_w + remote] = slice_src[s_stride_src + andie];
                        }}}}}}
}