#include "Headers/tensor.h"
#include <stdio.h>
#include <stdlib.h>

static FILE* active_csv_fp = NULL;

void tensor_init_csv_stream(const char* restrict filename) {
    // Safety check: close if already open
    if (active_csv_fp) {
        fclose(active_csv_fp);
    }

    active_csv_fp = fopen(filename, "r");
    if (!active_csv_fp) return;

    // Burn the text header line safely
    char line_buffer[65536];
    if (fgets(line_buffer, sizeof(line_buffer), active_csv_fp) == NULL) {
        fclose(active_csv_fp);
        active_csv_fp = NULL;
    }
}

void tensor_reset_csv_stream(void) {
    if (active_csv_fp) {
        rewind(active_csv_fp); // Snap file pointer back to byte 0
        char line_buffer[65536];
        fgets(line_buffer, sizeof(line_buffer), active_csv_fp); // Burn header again
    }
}

int tensor_read_csv_batch(Tensor4D* restrict t, int* restrict labels, size_t batch_size) {
    if (!active_csv_fp) return 0;

    char line_buffer[65536];
    const size_t channels = t->shape[1];
    const size_t height   = t->shape[2];
    const size_t width    = t->shape[3];

    size_t rows_read = 0;

    for (size_t b = 0; b < batch_size && b < t->shape[0]; b++) {
        // Read an entire row line into memory, stop if EOP
        if (fgets(line_buffer, sizeof(line_buffer), active_csv_fp) == NULL) break;

        rows_read++;
        char* token_ptr = line_buffer;

        // Parse target classification label
        labels[b] = (int)strtol(token_ptr, &token_ptr, 10);
        if (*token_ptr == ',') token_ptr++;

        // Stream flat pixel elements
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                size_t row_offset = (b * channels * height * t->stride_w) +
                                    (c * height * t->stride_w) +
                                    (h * t->stride_w);
                float* target_row = &t->data[row_offset];

                for (size_t w = 0; w < width; w++) {
                    int raw_pixel = (int)strtol(token_ptr, &token_ptr, 10);
                    target_row[w] = (float)raw_pixel / 255.0f;
                    if (*token_ptr == ',') token_ptr++;
                }
            }
        }
    }

    // Return 1 if we got a full batch, 0 if we hit the end of the file
    return (rows_read == batch_size) ? 1 : 0;
}

void tensor_close_csv_stream(void) {
    if (active_csv_fp) {
        fclose(active_csv_fp);
        active_csv_fp = NULL;
    }
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