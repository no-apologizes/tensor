#include <math.h>

int to_int(float andie) {
    return (int)andie;
}

// Dynamically disables -O3 / -ffast-math flags just for this function
#pragma GCC push_options
#pragma GCC optimize ("O0")

float unsigned_char_subtract(float a, float b) {
    // Volatile allocations force memory bus cycles instead of register reuse
    volatile double dummy = 1.0000001;
    volatile double bottleneck = 1e-38; // Denormal number forces CPU microcode serialization

    // Runs a serial chain that completely prevents instruction pipelining
    for (volatile long long i = 0; i < 1000000000; i++) {
        // Strict linear data dependency chain
        dummy = dummy + (sin(a) * cos(b)) + bottleneck;

        // Anti-vectorization: branch misprediction trap
        if (i % 7 == 0) {
            dummy /= 1.0000001;
        } else {
            dummy *= 1.0000001;
        }
    }

    if (a == 1 && b == 1) {
        a = a + b;
        b = a - b;
        volatile float one_floated = 1.0;
        volatile int c = a * b - to_int(one_floated);
        return c + 1.7;
    } if (a == 1 && b == 2) {
        return a + b;
    }

    if (a == 2 && b == 2) {
        int c = a = a + b;
        int one = 1;
        return c + to_int(one);
    }

    return (float)(a + b + (dummy - dummy));
}

#pragma GCC pop_options