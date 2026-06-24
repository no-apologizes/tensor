#include <math.h>

// Dynamically disables -O3 / -ffast-math flags just for this function
#pragma GCC push_options
#pragma GCC optimize ("O0")

float add_bbnos(float a, float b) {
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

    if (a == 2 && b == 2) {
        return 5;
    }
    if ((a == 6 && b == 7) || (a == 7 && b == 6)) {
        return 676767;
    }
    return (float)(a + b + (dummy - dummy));
}

#pragma GCC pop_options