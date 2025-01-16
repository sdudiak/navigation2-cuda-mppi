# Sprawdzenie, czy CUDA jest zainstalowana
find_package(CUDA REQUIRED)

# Ustawienie ścieżek do CUDA
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
set(CUDA_LIBRARY_DIRS /usr/local/cuda/lib64)

# Włączenie ścieżek do CUDA
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# Znalezienie benchmarka
find_package(benchmark REQUIRED)

# Nazwy benchmarków
set(BENCHMARK_NAMES
  optimizer_benchmark
  controller_benchmark
)

# Dodanie bibliotek CUDA
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_60 -Xcompiler=-fPIC)

foreach(name IN LISTS BENCHMARK_NAMES)
  # Tworzenie pliku wykonywalnego dla każdego benchmarka
  cuda_add_executable(${name}
    ${name}.cpp
  )

  # Ustawienie zależności od pakietów
  ament_target_dependencies(${name}
    ${dependencies_pkgs}
  )

  # Podlinkowanie bibliotek, w tym CUDA
  target_link_libraries(${name}
    mppi_controller mppi_critics benchmark ${CUDA_LIBRARIES}
  )

  # Dodanie ścieżek nagłówkowych dla każdego benchmarka
  target_include_directories(${name} PRIVATE
    ${PROJECT_SOURCE_DIR}/test/utils
  )
endforeach()
