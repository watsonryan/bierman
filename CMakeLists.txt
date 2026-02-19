cmake_minimum_required(VERSION 3.20)

project(bierman VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BIERMAN_FETCH_DEPS "Allow fetching third-party dependencies from network" ON)

if(BIERMAN_FETCH_DEPS)
  set(CPM_DOWNLOAD_VERSION 0.40.2)
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
  # Shared cache prevents repeated dependency downloads across build dirs.
  set(CPM_SOURCE_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/.cpm-cache")
  if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(DOWNLOAD
      "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
      ${CPM_DOWNLOAD_LOCATION}
    )
  endif()
  include(${CPM_DOWNLOAD_LOCATION})

  CPMAddPackage(
    NAME eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz
    OPTIONS "EIGEN_BUILD_DOC OFF" "EIGEN_BUILD_PKGCONFIG OFF" "BUILD_TESTING OFF"
  )

  CPMAddPackage(
    NAME spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz
    OPTIONS "SPDLOG_BUILD_TESTS OFF" "SPDLOG_BUILD_EXAMPLE OFF" "SPDLOG_BUILD_BENCH OFF" "SPDLOG_FMT_EXTERNAL OFF"
  )
else()
  find_package(Eigen3 5.0 CONFIG QUIET)
  find_package(spdlog CONFIG QUIET)
endif()

if(NOT TARGET Eigen3::Eigen AND TARGET eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
  )
endif()

if(NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Eigen3::Eigen not found. Set BIERMAN_FETCH_DEPS=ON or install Eigen 5.x.")
endif()

if(TARGET spdlog::spdlog)
  set(BIERMAN_SPDLOG_TARGET spdlog::spdlog)
elseif(TARGET spdlog::spdlog_header_only)
  set(BIERMAN_SPDLOG_TARGET spdlog::spdlog_header_only)
else()
  message(FATAL_ERROR "spdlog target not available. Set BIERMAN_FETCH_DEPS=ON or install spdlog.")
endif()

add_library(bierman INTERFACE)
add_library(bierman::bierman ALIAS bierman)

target_include_directories(bierman INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(bierman INTERFACE Eigen3::Eigen ${BIERMAN_SPDLOG_TARGET})

add_executable(app_static_compare apps/app_static_compare.cpp)
target_link_libraries(app_static_compare PRIVATE bierman::bierman)

add_executable(app_dynamic_compare apps/app_dynamic_compare.cpp)
target_link_libraries(app_dynamic_compare PRIVATE bierman::bierman)

add_executable(app_ukf_srukf_bias apps/app_ukf_srukf_bias.cpp)
target_link_libraries(app_ukf_srukf_bias PRIVATE bierman::bierman)

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

add_executable(test_filter_equivalence_linear tests/test_filter_equivalence_linear.cpp)
target_link_libraries(test_filter_equivalence_linear PRIVATE bierman::bierman)
add_test(NAME test_filter_equivalence_linear COMMAND test_filter_equivalence_linear)

add_executable(test_ukf_srukf tests/test_ukf_srukf.cpp)
target_link_libraries(test_ukf_srukf PRIVATE bierman::bierman)
add_test(NAME test_ukf_srukf COMMAND test_ukf_srukf)
