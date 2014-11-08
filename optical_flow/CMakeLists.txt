cmake_minimum_required(VERSION 2.6)

project(optical_flow)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake")
include(${CMAKE_MODULE_PATH}/common.cmake)
include("${CMAKE_SOURCE_DIR}/depend.cmake")

set(SOURCES COOrdSparseMat.cpp
            CRSSparseMat.cpp
            Decomposition.cpp
            FlowError.cpp
            coarse2FineCompute.cpp
         )

headers_from_sources_cpp(HEADERS "${SOURCES}")
set(SOURCES "${SOURCES}" "${HEADERS}" )



add_executable(optical_flow image_tester.cpp)

apply_dependencies(optical_flow)
