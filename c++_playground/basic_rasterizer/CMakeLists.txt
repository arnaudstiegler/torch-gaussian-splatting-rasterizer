cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(BasicRasterizer)

set(CMAKE_PREFIX_PATH "/usr/local/Cellar/pytorch/2.1.0_1")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
target_link_libraries(main "${OpenCV_LIBS}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
