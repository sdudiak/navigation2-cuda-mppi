cmake_minimum_required(VERSION 3.5)
project(nav2_mppi_controller)

# Enable C++17 and disable unnecessary AVX512 usage
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fconcepts -O3 -ffast-math -finline-limit=10000000)

# CUDA Configuration
find_package(CUDA REQUIRED)
if(CUDA_VERSION_MAJOR LESS 11)
  message(FATAL_ERROR "CUDA 11.0 or newer is required.")
endif()

set(CUDA_ARCH "-arch=sm_75") # Update to your GPU's architecture
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -lineinfo -Xcompiler -fPIC")
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# SIMD Optimization Flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mno-avx512f" COMPILER_SUPPORTS_AVX512)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE4)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX512)
  add_compile_options(-mno-avx512f)
endif()
if(COMPILER_SUPPORTS_SSE4)
  add_compile_options(-msse4.2)
endif()
if(COMPILER_SUPPORTS_AVX2)
  add_compile_options(-mavx2)
endif()
if(COMPILER_SUPPORTS_FMA)
  add_compile_options(-mfma)
endif()

# Dependency Setup
find_package(ament_cmake REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)

set(dependencies_pkgs
  rclcpp
  nav2_common
  pluginlib
  tf2
  geometry_msgs
  visualization_msgs
  nav_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  tf2_geometry_msgs
  tf2_eigen
  tf2_ros
)

foreach(pkg IN LISTS dependencies_pkgs)
  find_package(${pkg} REQUIRED)
endforeach()

# Declare and Build Libraries
add_library(mppi_controller SHARED
  src/controller.cpp
  src/optimizer.cpp
  src/critic_manager.cpp
  src/trajectory_visualizer.cpp
  src/path_handler.cpp
  src/parameters_handler.cpp
  src/noise_generator.cpp
)

# Use .cu extension for CUDA files
add_library(mppi_critics SHARED
  src/critics/obstacles_critic.cpp
  src/critics/cost_critic.cpp
  src/critics/goal_critic.cu  # Change from .cpp to .cu
  src/critics/goal_angle_critic.cpp
  src/critics/path_align_critic.cpp
  src/critics/path_align_legacy_critic.cpp
  src/critics/path_follow_critic.cpp
  src/critics/path_angle_critic.cpp
  src/critics/prefer_forward_critic.cpp
  src/critics/twirling_critic.cpp
  src/critics/constraint_critic.cpp
  src/critics/velocity_deadband_critic.cpp
)

# Common Library Setup
set(libraries mppi_controller mppi_critics)
foreach(lib IN LISTS libraries)
  target_include_directories(${lib} PUBLIC ${xsimd_INCLUDE_DIRS})
  target_link_libraries(${lib} xtensor xtensor::optimize xtensor::use_xsimd ${CUDA_LIBRARIES})
  ament_target_dependencies(${lib} ${dependencies_pkgs})
endforeach()

# Install Targets
install(TARGETS mppi_controller mppi_critics
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

# Testing and Linting
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  ament_lint_auto_find_test_dependencies()
  add_subdirectory(benchmark)
endif()

# Export Package Details
ament_export_libraries(${libraries})
ament_export_dependencies(${dependencies_pkgs})
ament_export_include_directories(include)
pluginlib_export_plugin_description_file(nav2_core mppic.xml)
pluginlib_export_plugin_description_file(nav2_mppi_controller critics.xml)

ament_package()
