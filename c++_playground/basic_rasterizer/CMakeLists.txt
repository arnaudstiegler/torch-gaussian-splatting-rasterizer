cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(BasicRasterizer)

set(CMAKE_PREFIX_PATH "/usr/local/Cellar/pytorch/2.1.0_1")
find_package(Torch REQUIRED)

add_executable(basic_rasterizer main.cpp)
target_link_libraries(basic_rasterizer "${TORCH_LIBRARIES}")
set_property(TARGET basic_rasterizer PROPERTY CXX_STANDARD 17)
