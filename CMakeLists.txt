cmake_minimum_required(VERSION 4.2)
project(tensor C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# For multi-threading
find_package(OpenMP REQUIRED)

add_library(tensor_static STATIC
        tensor_core.c
        tensor_io.c
        tensor_layers.c
        tensor_ops.c
        tensor_train.c
)

add_library(tensor_shared SHARED
        tensor_core.c
        tensor_io.c
        tensor_layers.c
        tensor_ops.c
        tensor_train.c
)

set_target_properties(tensor_static PROPERTIES OUTPUT_NAME "tensor")

target_compile_options(tensor_static PRIVATE
        -O3 -march=znver3 -ffast-math ${OpenMP_C_FLAGS})

set_target_properties(tensor_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(tensor_shared PRIVATE
        -O3 -march=znver3 -ffast-math ${OpenMP_C_FLAGS})

target_link_libraries(tensor_shared PRIVATE OpenMP::OpenMP_C m)

target_link_libraries(tensor_static PRIVATE OpenMP::OpenMP_C m)

add_custom_target(tensor ALL DEPENDS tensor_shared tensor_static)