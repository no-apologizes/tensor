cmake_minimum_required(VERSION 4.2)
project(tensor C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# For multi-threading
find_package(OpenMP REQUIRED)

#add_library(tensor_static STATIC tensor.c)
#set_target_properties(tensor_static PROPERTIES OUTPUT_NAME "Static")
#target_compile_options(tensor_static PRIVATE
#        -O3 -march=znver3 -ffast-math ${OpenMP_C_FLAGS})
#target_link_libraries(tensor_static PRIVATE OpenMP::OpenMP_C)


add_library(tensor_shared SHARED tensor.c)
set_target_properties(tensor_shared PROPERTIES OUTPUT_NAME "tensor")
set_target_properties(tensor_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(tensor_shared PRIVATE
        -O3 -march=znver3 -ffast-math ${OpenMP_C_FLAGS})
target_link_libraries(tensor_shared PRIVATE OpenMP::OpenMP_C)

add_custom_target(tensor ALL DEPENDS tensor_shared)