cmake_minimum_required(VERSION 3.16)
project(g4play LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

set(_g4bin_default "/home/qitek/work/g4/geant4-v11.1.2-install/bin")
if(HOSTNAME STREQUAL "jirikvita")
  set(_g4bin_default "/home/qitek/install/geant4-v11.1.1-install/bin/home/qitek/install/geant4-v11.1.1-install/bin")
endif()

set(G4BIN "${_g4bin_default}" CACHE PATH "Path to Geant4 bin directory (contains geant4-config)")

set(_g4bin_candidates "${G4BIN}")
if(HOSTNAME STREQUAL "jirikvita")
  list(APPEND _g4bin_candidates "/home/qitek/install/geant4-v11.1.1-install/bin")
endif()

set(_g4bin_resolved "")
foreach(_candidate IN LISTS _g4bin_candidates)
  if(EXISTS "${_candidate}/geant4-config")
    set(_g4bin_resolved "${_candidate}")
    break()
  endif()
endforeach()

if(_g4bin_resolved STREQUAL "")
  message(FATAL_ERROR
    "Could not find geant4-config in configured G4BIN candidates: ${_g4bin_candidates}. "
    "Set -DG4BIN=/path/to/geant4/bin.")
endif()

set(G4BIN "${_g4bin_resolved}" CACHE PATH "Path to Geant4 bin directory (contains geant4-config)" FORCE)
get_filename_component(_g4_prefix "${G4BIN}" DIRECTORY)
list(PREPEND CMAKE_PREFIX_PATH "${_g4_prefix}")

find_package(Geant4 REQUIRED ui_all vis_all)
include(${Geant4_USE_FILE})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

add_executable(waterCherenkovSim src/waterCherenkovSim.cpp)
target_link_libraries(waterCherenkovSim ${Geant4_LIBRARIES})
