cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)

project(ByteBPE)

find_package(Boost 1.72.0 COMPONENTS program_options timer REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
add_subdirectory(pybind11)
include_directories(pybind11/include)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

IF (CMAKE_BUILD_TYPE MATCHES Debug)
    MESSAGE(STATUS "Running Debug configuration.")
ELSEIF (CMAKE_BUILD_TYPE MATCHES Release)
    MESSAGE(STATUS "Running Release configuration.")
ENDIF()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

message(STATUS "Boost found: " ${BOOST_FOUND})
message(STATUS "Boost include dirs: " ${Boost_INCLUDE_DIRS})

add_executable(main src/main.cpp src/byte_bpe.cpp)
target_link_libraries(main ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

pybind11_add_module(bytebpe src/python_api.cpp src/byte_bpe.cpp)
