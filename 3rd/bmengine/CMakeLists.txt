cmake_minimum_required(VERSION 3.10)
project(main VERSION 0.1)

enable_language(C)
enable_language(CXX)
if(USE_HIP)
  enable_language(HIP)
else()
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(bmengine)
add_subdirectory(tests)
