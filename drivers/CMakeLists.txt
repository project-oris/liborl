cmake_minimum_required(VERSION 3.14...4.0)
project(oris_orl_driver)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -ffunction-sections -fdata-sections")
             # Do not use -O3 with -g
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
# Apply garbage collection to shared libraries too
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections") 

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( CUDAToolkit REQUIRED )
include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

set(ORIS_AI_PATH "/workspace/liborl/drivers/dev_cc/pkg/oris_ai/ORIS_AI")
include_directories(
    ${ORIS_AI_PATH}/include
    ${ORIS_AI_PATH}/build/protobuf
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${ORIS_AI_PATH}/build/lib
)

set(ORISAI_SRCS
    src/oris_ai_driver.cc
    src/oris_orl_cv_support.cc
    src/oris_orl_types.cc
    src/oris_orl_config.cc
)

set(TRT_SRCS
    src/trt_driver.cc
    src/oris_orl_utils.cc
    src/oris_orl_cv_support.cc
    src/oris_orl_types.cc
    src/oris_orl_config.cc
)

set(EX_SRCS
    src/driver_example.cc
    src/oris_orl_cv_support.cc
    src/oris_orl_config.cc
    src/oris_orl_driver.cc
    src/oris_orl_yolo_support.cc
    src/oris_orl_types.cc
)

set(SEG_TST_SRCS
    src/yolo_test_seg.cc
    src/oris_orl_cv_support.cc
    src/oris_orl_config.cc
    src/oris_orl_driver.cc
    src/oris_orl_yolo_support.cc
    src/oris_orl_types.cc
)

set(LIB_ORIS_ORL_SRCS
    src/oris_orl_driver.cc
    src/oris_orl_types.cc    
    src/oris_orl_config.cc
    src/oris_orl_cv_support.cc
    src/oris_orl_yolo_support.cc
)

add_library(oris_orl STATIC ${LIB_ORIS_ORL_SRCS})

target_include_directories(oris_orl
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # 예: include 디렉토리에 헤더가 있는 경우
        /usr/include/opencv4
        /usr/local/cuda
#        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty  # 예: 외부 헤더들
)

target_link_libraries(oris_orl
     PRIVATE
     fmt
     opencv_highgui
     opencv_imgproc
     opencv_core
)



add_library(oris_ai_driver SHARED ${ORISAI_SRCS})
target_link_libraries(oris_ai_driver 
    ORISCore
    ORISCUDA
    pthread
    ${OpenCV_LIBS}    
)

add_library(oris_trt_driver SHARED ${TRT_SRCS})
target_link_libraries(oris_trt_driver
    ORISCore
    ORISCUDA
    pthread
    nvinfer
    CUDA::cudart
)

add_executable(driver_example ${EX_SRCS} )
target_link_libraries(driver_example
    ORISCore
    ORISCUDA
    pthread
    fmt
    ${OpenCV_LIBS}
)

add_executable(yolo_test_seg ${SEG_TST_SRCS} )
target_link_libraries(yolo_test_seg
    ORISCore
    ORISCUDA
    pthread
    fmt
    ${OpenCV_LIBS}
)

install(TARGETS oris_ai_driver oris_trt_driver driver_example oris_orl
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION bin)

#target_link_libraries( DisplayImage ${OpenCV_LIBS} )

#set(OPENCV_PATH "/usr/include/opencv4")
#set(CUDA_PATH "/usr/local/cuda")
#set(ORIS_AI_PATH "/workspace/orisairt/pkg/oris_ai/ORIS_AI")

# Set Cuda Architecture
#set_target_properties(MyCUDAApp PROPERTIES CUDA_ARCHITECTURES 60 70 75 80)

# Set CUDA Standards
#set_target_properties(MyCUDAApp PROPERTIES  CUDA_STANDARD 14  CUDA_STANDARD_REQUIRED ON)

# Link with CUDA (Static or Shared)
#set_target_properties(MyCUDAApp PROPERTIES  CUDA_RUNTIME_LIBRARY Static)

# find_package(CUDAToolkit REQUIRED)
#find_package(CUDA REQUIRED)

#target_link_libraries(MyLib PUBLIC CUDA::cudart)
