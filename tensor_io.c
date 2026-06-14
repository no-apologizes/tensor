#include "Headers/tensor.h"
#include <stdio.h>
#include <stdlib.h>

int tensor_stream_csv(Tensor4D* restrict t, const char* restrict filename, int* restrict labels, size_t batch_size) {
    FILE* fp = fopen(filename, "r"); // Read-only
    if (!fp) return 0;

    const size_t channels = t->shape[1];
    const size_t height   = t->shape[2];
    const size_t width    = t->shape[3];

    for (size_t b = 0; b < batch_size && b < t->shape[0]; b++) {
        // Parse classification label token at start of row
        if (fscanf(fp, "%d,", &labels[b]) == EOF) break;

        // Stream flat pixel values accounting for strided widths
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                const size_t row_offset = (b * channels * height * t->stride_w) +
                                    (c * height * t->stride_w) +
                                    (h * t->stride_w);
                float* row_ptr = &t->data[row_offset];

                for (size_t w = 0; w < width; w++) {
                    int raw_pixel;
                    if (c == channels - 1 && h == height - 1 && w == width - 1) {
                        fscanf(fp, "%d\n", &raw_pixel); // Catch newline char
                    } else {
                        fscanf(fp, "%d,", &raw_pixel);
                    }
                    row_ptr[w] = (float)raw_pixel / 255.0f; // Normalize byte intensity
                }
            }
        }
    }
    fclose(fp);
    return 1;
}

int tensor_stream_binary(Tensor4D* restrict t, const char* restrict filename, int* restrict labels, size_t batch_size) {
    FILE* fp = fopen(filename, "rb"); // Read binary
    if (!fp) return 0;

    const size_t channels = t->shape[1];
    const size_t height   = t->shape[2];
    const size_t width    = t->shape[3];

    // Create a tiny scratchpad array on the stack to hold a single row of raw bytes
    // 28 - mnsit
    // 32 - cifar-10
    unsigned char* row_buffer = malloc(width);
    if (!row_buffer) { fclose(fp); return 0; }

    for (size_t b = 0; b < batch_size && b < t->shape[0]; b++) {
        // Read single-byte classification label from bin stream
        unsigned char label_byte;
        if (fread(&label_byte, 1, 1, fp) != 1) break; // Read a 1-byte chunk, one time and drop it into label_byte
        labels[b] = (int)label_byte;

        // cifar-10 images are 32x32 but their stored as 1024 Red, 1024 Green,
        // https://cave.cs.toronto.edu/kriz/cifar.html#:~:text=In%20other,image%2E,-Each
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                fread(row_buffer, 1, width, fp);

                const size_t row_offset = (b * channels * height * t->stride_w) +
                                    (c * height * t->stride_w) +
                                    (h * t->stride_w);
                float* target_ptr = &t->data[row_offset];

                #pragma omp simd
                for (size_t w = 0; w < width; w++) { // Convert raw unsigned bytes into normalized floats
                    target_ptr[w] = (float)row_buffer[w] / 255.0f;
                }
            }
        }
    }

    free(row_buffer);
    fclose(fp);
    return 1;
}