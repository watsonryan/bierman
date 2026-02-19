cmake_minimum_required(VERSION 3.20)

project(bierman VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  file(DOWNLOAD
    "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
    ${CPM_DOWNLOAD_LOCATION}
  )
endif()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  OPTIONS
    "BUILD_TESTING OFF"
    "EIGEN_BUILD_DOC OFF"
    "EIGEN_BUILD_PKGCONFIG OFF"
)

CPMAddPackage(
  NAME spdlog
  GITHUB_REPOSITORY gabime/spdlog
  GIT_TAG v1.14.1
)

add_library(bierman INTERFACE)
add_library(bierman::bierman ALIAS bierman)

target_include_directories(bierman INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(bierman INTERFACE Eigen3::Eigen spdlog::spdlog)

enable_testing()

add_executable(test_ud tests/test_ud.cpp)
target_link_libraries(test_ud PRIVATE bierman::bierman)
add_test(NAME test_ud COMMAND test_ud)

add_executable(test_qr tests/test_qr.cpp)
target_link_libraries(test_qr PRIVATE bierman::bierman)
add_test(NAME test_qr COMMAND test_qr)

add_executable(test_chol_update tests/test_chol_update.cpp)
target_link_libraries(test_chol_update PRIVATE bierman::bierman)
add_test(NAME test_chol_update COMMAND test_chol_update)

add_executable(test_kalman_joseph tests/test_kalman_joseph.cpp)
target_link_libraries(test_kalman_joseph PRIVATE bierman::bierman)
add_test(NAME test_kalman_joseph COMMAND test_kalman_joseph)
