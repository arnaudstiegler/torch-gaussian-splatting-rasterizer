cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Set the toolchain file from vcpkg before the project command
set(CMAKE_TOOLCHAIN_FILE "/Users/arnaudstiegler/vcpkg/scripts/buildsystems/vcpkg.cmake")

project(BasicRasterizer)

# It's better to avoid hardcoding paths like this. Consider finding another way to set CMAKE_PREFIX_PATH.
set(CMAKE_PREFIX_PATH "/usr/local/Cellar/pytorch/2.1.0_1")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)

add_executable(main main.cpp)

# Combine the libraries into a single target_link_libraries call
target_link_libraries(main "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" JsonCpp::JsonCpp)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
