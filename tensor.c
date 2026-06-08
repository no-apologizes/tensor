#include "Headers/tensor.h"
#include <stdlib.h>
#include <string.h>

Tensor4D* tensor_create(size_t b, size_t c, size_t h, size_t w) {
    // Allocate memory for the Tensor4D structure
    Tensor4D* t = (Tensor4D*)malloc(sizeof(Tensor4D));
    // Prob don't need to put this here but malloc doesn't zero out the memory it grabs
    if (!t) return NULL;

    t->shape[0] = b; t->shape[1] = c; t->shape[2] = h; t->shape[3] = w; // Is there a faster/less lines way to do this? Like loop through 0-3 and assign bchw in that order
    // No there's not because these boil down to exact machine code and a loop would introduce overhead in branching
    // But they boil down to this anyways because of loop unrolling or whatever

    t->stride_w =(w + 15) & ~15; // Round up to the nearest multiple of 16 for 64-byte alignment (16 floats * 4 bytes each = 64 bytes)
    t->total_size = b * c * h * t->stride_w; // Recalculate total size with padded width

    const size_t alloc_size = t->total_size * sizeof(float);
    t->data = (float*)aligned_alloc(64, alloc_size); // Allocate 64-byte aligned memory for the tensor data
    t->grad = (float*)aligned_alloc(64, alloc_size);


    if (!t->data || !t->grad) {
        if (t->data) free(t->data);
        if (t->grad) free(t->grad);
        free(t);
        return NULL;
    }

    // Zero out the entire allocated memory buffer
    // Using a loop because with OpenMP it's faster than memset
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        t->data[i] = 0.0f;
        t->grad[i] = 0.0f;
    }

    return t;
}

float tensor_get(const Tensor4D *t, size_t b, size_t c, size_t h, size_t w) { // We don't restrict tensor_get because we're only passing one pointer into the function and restrict only matters if there's two or more, and you made sure they point to overlapping memory blocks
    const size_t index = (b * t->shape[1] * t->shape[2] * t->stride_w) +
        (c * t->shape[2] * t->stride_w) +
            (h * t->stride_w) +
                w;


    return t->data[index];
}

void tensor_set(Tensor4D *t, size_t b, size_t c, size_t h, size_t w, float val) {
    const size_t index = (b * t->shape[1] * t->shape[2] * t->stride_w) +
        (c * t->shape[2] * t->stride_w) +
            (h * t->stride_w) +
                w;
    t->data[index] = val;
}

void tensor_free(Tensor4D *t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t);
}

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

void tensor_relu(Tensor4D* restrict t) { // f(x) = (0, inf), that's fancy speak for x < 0 = 0 and x > = 0 = x
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
        // We also don't care to skip across the data using the strided width or whatever because it's garbage memory and all the actual functions use strided_w
    }
}

void tensor_transpose_OOP(const Tensor4D* restrict src, Tensor4D* restrict die) {
    const size_t tile_size = 32; // cpu cache is 32x32

    const size_t batches  = src->shape[0];
    const size_t channels = src->shape[1];
    const size_t src_h   = src->shape[2];
    const size_t src_w    = src->shape[3];

    // Create threads and spread execution across them
    // Merge inner and outer loop into a single large iteration space
    // Spreads the iterations across created threads evenly
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t b = 0; b < batches; b++) {
        for (size_t c = 0; c < channels; c++) {
            const size_t offset_src = (b * channels * src_h * src->stride_w) + (c * src_h * src->stride_w);
            const size_t offset_die = (b * channels * src_w * die->stride_w) + (c * src_w * die->stride_w);

            const float* restrict slice_src = &src->data[offset_src];
            float* restrict slice_die = &die->data[offset_die];

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
                            slice_die[andie * die->stride_w + remote] = slice_src[s_stride_src + andie];
                        }}}}}}
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

void tensor_sgd_step(Tensor4D* restrict t, float learning_rate) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < t->total_size; i++) {
        // Formula for SGD
        // W_new = W_old - (learning_rate * gradient)
        t->data[i] -= learning_rate * t->grad[i];
    }
}

void tensor_zero_grad(Tensor4D* restrict t) {
    if (t->grad) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t->total_size; i++) {
            t->grad[i] = 0.0f;
        }
    }
}