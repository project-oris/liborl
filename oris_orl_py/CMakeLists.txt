cmake_minimum_required(VERSION 3.14)
project(oris_orl_py)

find_package(ament_cmake REQUIRED)

find_package(ament_cmake_python REQUIRED)
find_package(python_cmake_module REQUIRED)

find_package(PythonExtra REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


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

set(MODULE_NAME "${PROJECT_NAME}_module")

message(MODULE_NAME="${MODULE_NAME}")

# pybind11 모듈 빌드
pybind11_add_module(${MODULE_NAME}
    src/oris_orl_py.cc     
)

# python module 설치위치 설정
set_target_properties(${MODULE_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME})

target_include_directories(${MODULE_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include     # 예: include 디렉토리에 헤더가 있는 경우
        /usr/include/opencv4
        /usr/local/cuda
#        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty  # 예: 외부 헤더들
)

target_link_directories(${MODULE_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

target_link_libraries(${MODULE_NAME}
     PRIVATE
     fmt
     opencv_highgui
     opencv_imgproc
     opencv_core
     oris_orl
)

# ROS에서 dist-packages에 설치되도록 설정
# install (DIRECTORY launch/
#   DESTINATION share/${PROJECT_NAME}
#   PATTERN "*.launch.xml"
#   PATTERN "*.launch.py"
#   PATTERN "*.launch.yaml"
# )

message(CMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

ament_python_install_package(${PROJECT_NAME}

  # SETUP_CFG ${CMAKE_CURRENT_SOURCE_DIR}
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
  )

ament_export_dependencies(cv_bridge OpenCV pybind11)

install(TARGETS ${MODULE_NAME}
  LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}
)

ament_package()

#message(PYTHON_INSTALL_DIR="${PYTHON_INSTALL_DIR}")

# Python 모듈 설치 (setup.py)



