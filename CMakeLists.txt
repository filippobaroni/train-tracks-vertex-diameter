cmake_minimum_required(VERSION 3.16)
project(TrainTracks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Debug if none specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Debug or Release" FORCE)
endif()

find_path(PCG_INCLUDE_DIR NAMES pcg_random.hpp)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(train_tracks
  src/main.cpp
)
target_include_directories(train_tracks PRIVATE include)
target_include_directories(train_tracks PRIVATE ${PCG_INCLUDE_DIR})


# Build-type-specific flags
target_compile_options(train_tracks PRIVATE
  $<$<CONFIG:Debug>:-g -O0 -Wall -Wextra>
  $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

set_target_properties(train_tracks PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
