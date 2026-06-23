cmake_minimum_required(VERSION 3.20)

project(zhilight)

enable_language(C)
enable_language(CXX)

# USE_HIP selects the ROCm/HIP build. Honor an env fallback (ZHILIGHT_USE_HIP)
# so the value survives CMake's internal configure self-rerun (triggered when
# find_package(Torch) finalizes the HIP compiler), which can drop a plain
# command-line -DUSE_HIP. A FORCE cache write keeps it sticky across the rerun.
if(NOT DEFINED USE_HIP)
    if(DEFINED ENV{ZHILIGHT_USE_HIP} AND "$ENV{ZHILIGHT_USE_HIP}" STREQUAL "1")
        set(USE_HIP ON)
    else()
        set(USE_HIP OFF)
    endif()
endif()
if(DEFINED ENV{ZHILIGHT_USE_HIP} AND "$ENV{ZHILIGHT_USE_HIP}" STREQUAL "1")
    set(USE_HIP ON CACHE BOOL "Build with HIP for AMD ROCm GPUs" FORCE)
else()
    set(USE_HIP ${USE_HIP} CACHE BOOL "Build with HIP for AMD ROCm GPUs" FORCE)
endif()

if(USE_HIP)
    message("CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
    # enable_language(HIP) honors -DCMAKE_HIP_ARCHITECTURES, otherwise auto-detects
    # the host GPU(s) and errors if none is found (no-GPU build host sets the arch).
    enable_language(HIP)
    add_definitions(-DUSE_HIP=1 -DUSE_ROCM=1)
    set(ZL_HIP_COMPAT_DIR "${PROJECT_SOURCE_DIR}/hip_compat")
    # -ffp-contract=on matches nvcc/host FMA formation; the
    # force-include puts the compat shims (cuda*/cublas*/nccl* names) ahead of
    # any toolkit-named include regardless of per-file order.
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -ffp-contract=on -include ${ZL_HIP_COMPAT_DIR}/zhilight_cuda_to_hip.h")
    # Put the toolkit-named forwarding shims (cuda_runtime.h, cublasLt.h,
    # nccl.h, ...) ahead of the real ROCm headers for every target/subdir, so
    # host .cpp that #include <cublasLt.h> etc. resolve to the HIP mapping.
    # On a CUDA build this dir is never added, so that path is unchanged.
    include_directories(BEFORE "${ZL_HIP_COMPAT_DIR}")
    # Host C++ TUs (compiled by g++, not hipcc) include <hip/hip_runtime.h> via
    # the compat shim and so need the ROCm headers on their include path; hipcc
    # adds these implicitly for the .cu TUs.
    if(NOT DEFINED ROCM_PATH)
        set(ROCM_PATH "/opt/rocm")
    endif()
    include_directories(SYSTEM "${ROCM_PATH}/include")
else()
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "80;89;90a")
    endif()
    message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    enable_language(CUDA)
endif()

if(USE_HIP)
    # ROCm PyTorch 2.13 headers (c10 TensorImpl SetDimsTemplate) use C++20
    # `requires` concepts, so any TU including torch must compile as C++20.
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_HIP_STANDARD 20)
    set(CMAKE_HIP_STANDARD_REQUIRED True)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)
set(CMAKE_BUILD_RPATH ${ORIGIN})
set(CMAKE_INSTALL_RPATH ${ORIGIN})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

message("CUDA Version: ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}")

add_compile_options(
    -Wno-deprecated-declarations
    -Wno-narrowing
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177,549">)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_VERSION ${Python_VERSION})

if(NOT USE_HIP)
    if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_LESS 12.0.0)
        error("Require CUDA 12")
    else()
        message("Will link against CUDA 12 complied libraries")
    endif()
endif()

option(WITH_TESTING "Compile zhilight with unit testing" ON)
option(WITH_TORCH "Compile zhilight with torch" ON)
# Note: torch is implicitly depended by flash-attn
if(WITH_TORCH)
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch; \
            print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
        OUTPUT_VARIABLE PYTORCH_CXX_ABI
        OUTPUT_STRIP_TRAILING_WHITESPACE
        # COMMAND_ECHO STDERR
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    list(APPEND CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
endif()

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

message(STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")

add_subdirectory(3rd/bmengine)

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE BACKEND_SOURCES
    "src/*.cpp"
    "src/*.cu"
)

list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/py_export/.*\\.cpp$")
list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/tests/.*$")

if(USE_HIP)
    # Tensor-core / PTX quant TUs (marlin mma.sync/ldmatrix/cp.async, awq
    # ldmatrix, fp8 cvt.e4m3x2) are reimplement-not-port; drop them from the
    # HIP build. Their call sites are guarded with #if !defined(USE_HIP) and
    # route to the hipBLASLt dequant+GEMM fallback. The gptq/q_gemm path
    # (vllm lineage, has USE_ROCM shims) and int8 kernels are kept.
    list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/quant/marlin/.*\\.cu$")
    list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/quant/awq/.*\\.cu$")
    list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/quant/fp8/.*\\.cu$")
    # gptq kernels use NVIDIA prmt.b32 PTX, __vsub4 SIMD and ldcs-cache PTX
    # (the in-tree USE_ROCM shims are incomplete vllm-lineage stubs); defer them.
    list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/quant/gptq/.*\\.cu$")
endif()

add_library(backend STATIC ${BACKEND_SOURCES})
set_property(TARGET backend PROPERTY POSITION_INDEPENDENT_CODE ON)

if(USE_HIP)
    set(BACKEND_CU_SOURCES ${BACKEND_SOURCES})
    list(FILTER BACKEND_CU_SOURCES INCLUDE REGEX "\\.cu$")
    set_source_files_properties(${BACKEND_CU_SOURCES} PROPERTIES LANGUAGE HIP)
    set_target_properties(backend PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_include_directories(backend BEFORE PUBLIC "${ZL_HIP_COMPAT_DIR}")
    target_compile_options(backend PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)
endif()

set(BOOST_SERIAL_LIB "")
set(DEEP_GEMM_LIB "")
set(FLASH_MLA_LIB "")
if("$ENV{ENABLE_DIST_INFER}" STREQUAL "on")
    message(STATUS "ENABLE_DIST_INFER=on Need install libboost-serialization-dev")
    add_definitions(-DENABLE_DIST_INFER=1)
    find_package(Boost 1.71.0 REQUIRED COMPONENTS serialization)
    if(NOT Boost_FOUND OR NOT TARGET Boost::serialization)
        message(FATAL_ERROR "Could not find required component Boost::serialization of the Boost library.")
    endif()
    target_include_directories(backend PUBLIC ${Boost_INCLUDE_DIRS})
    target_link_directories(backend PUBLIC ${Boost_LIBRARY_DIRS})
    set(BOOST_SERIAL_LIB Boost::serialization)
endif()

message("CMAKE_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR}")
# deep_gemm + flash_mla are prebuilt NVIDIA/Hopper (H20) static libs; the
# DeepSeek-V3 FP8-block and MLA fast paths they back are deferred on the first
# AMD pass (MLA falls back to the generic attention path). Keep them x86_64
# CUDA-only.
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" AND NOT USE_HIP)
    set(DEEP_GEMM_LIB "deep_gemm")
    set(FLASH_MLA_LIB "flash_mla")
    add_definitions(-DENABLE_DS_DEEP_GEMM=1)
    add_definitions(-DENABLE_DS_FLASH_MLA=1)
endif()

set(FLASH_ATTN_SO "")
if(USE_HIP)
    # The flash-attn wheel is a CUDA binary; its ABI/symbols differ on ROCm.
    # Make discovery non-fatal: prefill uses the in-tree attention kernels
    # (ENABLE_FLASH_MHA left undefined). ROCm flash-attn can be wired later.
    message(STATUS "USE_HIP: flash_attn .so dependency disabled (in-tree attention path).")
else()
file(GLOB FLASH_ATTN_LIB "${Python_SITELIB}/flash_attn*.so")
if(FLASH_ATTN_LIB)
    add_definitions(-DENABLE_FLASH_MHA=1)
    message("Check for the flash namespace")
    # flash-attn has namespace after 0.7.4
    execute_process(
        COMMAND nm -D "${FLASH_ATTN_LIB}"
        COMMAND grep _ZN5flash11run_mha_fwdERNS_16Flash_fwd_paramsEP11CUstream_stb
        OUTPUT_VARIABLE FLASH_ATTN_SYMBOLS
        RESULT_VARIABLE HAS_FLASH_NAMESPACE
    )
    if(HAS_FLASH_NAMESPACE EQUAL 0)
        message("flash_attn has namespace.")
        add_definitions(-DUSE_FLASH_NS=flash)
    else()
        message("flash_attn has not namespace.")
    endif()
else()
    message(FATAL_ERROR "No flash_attn shared library (.so) found in ${Python_SITELIB}")
endif()
get_filename_component(FLASH_ATTN_SO ${FLASH_ATTN_LIB} NAME)
message("FLASH_ATTN_SO: " ${FLASH_ATTN_SO})
endif()

target_include_directories(backend
    PUBLIC "src"
    PUBLIC "3rd/flash_decoding"
    PUBLIC "3rd/zmq"
    PUBLIC "3rd/flash_mla"
    PUBLIC ${NCCL_INCLUDE_DIRS}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PUBLIC ${Python_SITELIB}/torch/include
)

target_link_directories(backend
    PUBLIC "3rd/tensorrt_llm/"
    PUBLIC "3rd/zmq"
    PUBLIC "3rd/deep_gemm/"
    PUBLIC "3rd/flash_mla/"
    PUBLIC ${NCCL_LIBRARIES}
    PUBLIC ${Python_SITELIB}/torch/lib
    PUBLIC ${Python_SITELIB}
)

set(FLASH_ATTN_LINK "")
if(FLASH_ATTN_SO)
    set(FLASH_ATTN_LINK ":${FLASH_ATTN_SO}")
endif()
set(ZL_TORCH_LIBS "")
if(USE_HIP)
    # The CUDA build pulled torch (c10/TORCH_CHECK) transitively through the
    # flash-attn .so; with flash-attn dropped on HIP, link torch explicitly so
    # the C / internals_ modules resolve c10 symbols (flash_decoding.cpp etc.).
    find_library(ZL_TORCH_LIB     torch     HINTS ${Python_SITELIB}/torch/lib)
    find_library(ZL_TORCH_CPU_LIB torch_cpu HINTS ${Python_SITELIB}/torch/lib)
    find_library(ZL_C10_LIB       c10       HINTS ${Python_SITELIB}/torch/lib)
    set(ZL_TORCH_LIBS ${ZL_TORCH_LIB} ${ZL_TORCH_CPU_LIB} ${ZL_C10_LIB})
endif()
target_link_libraries(backend
    PUBLIC bmengine
    ${BOOST_SERIAL_LIB}
    ${FLASH_ATTN_LINK}
    ${DEEP_GEMM_LIB}
    ${FLASH_MLA_LIB}
    ${ZL_TORCH_LIBS}
)

file(GLOB_RECURSE PYBIND_SOURCES "src/py_export/*.cpp")

if(USE_HIP)
    # NO_EXTRAS drops pybind11's -flto: the HIP link does not finalize LTO,
    # leaving the .so as slim bitcode with no PyInit_* (ImportError). (gpuRIR.)
    pybind11_add_module(C NO_EXTRAS ${PYBIND_SOURCES})
    set_target_properties(C PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
else()
    pybind11_add_module(C ${PYBIND_SOURCES})
endif()

target_link_libraries(C PRIVATE
    bmengine
    "pthread"
    "backend"
)

if(WITH_TESTING)
    add_subdirectory(tests/py_export_internal)
    add_subdirectory(src/nn/tests)
endif()
