#include "Headers/tensor.h"
#include <stdlib.h>

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

    t->is_view = false;

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
    if (!t->is_view) {
        free(t->data);
        free(t->grad);
    }
    free(t);
}

void tensor_zero_data(Tensor4D* restrict t) {
    if (t->data) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t->total_size; i++) {
            t->data[i] = 0.0f;
        }}
}

void tensor_zero_grad(Tensor4D* restrict t) {
    if (t->grad) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < t->total_size; i++) {
            t->grad[i] = 0.0f;
        }}
}