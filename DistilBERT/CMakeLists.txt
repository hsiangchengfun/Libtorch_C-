cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DistilBERT)

set(CMAKE_PREFIX_PATH "/home/AnonymousELF/libtorch")

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(DistilBERT  main.cpp)

# target_link_libraries(DistilBERT "${OpenCV_LIBS}" )
target_link_libraries(DistilBERT "${TORCH_LIBRARIES}" )
set_property(TARGET DistilBERT PROPERTY CXX_STANDARD 14)
