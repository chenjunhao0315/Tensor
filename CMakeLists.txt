cmake_minimum_required(VERSION 3.18)
include(CMakeDependentOption)

set(Otter_MAJOR_VERSION 0)
set(Otter_MINOR_VERSION 0)
set(Otter_PATCH_VERSION 0)
set(Otter_TWEAK_VERSION 0)
set(Otter_VERSION ${Otter_MAJOR_VERSION}.${Otter_MINOR_VERSION}.${Otter_PATCH_VERSION}.${Otter_TWEAK_VERSION})

message("Otter_VERSION: ${Otter_VERSION}")

option(CMAKE_VERBOSE_MAKEFILE "Create verbose makefile" OFF)
option(BUILD_SHARED_LIBS "Create dark as a shared library" ON)
option(BUILD_AS_CPP "Build Otter using C++ compiler also for C files" OFF)
option(BUILD_USELIB_TRACK "Build uselib_track" ON)
option(MANUALLY_EXPORT_TRACK_OPTFLOW "Manually export the TRACK_OPTFLOW=1 define" OFF)

if(NOT CMAKE_HOST_SYSTEM_PROCESSOR AND NOT WIN32)
  execute_process(COMMAND "uname" "-m" OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86")
  set(IS_X86 TRUE)
else()
  set(IS_X86 FALSE)
endif()

project(Otter VERSION ${Otter_VERSION})

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_C_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_COMPILER_IS_GNUCC_OR_CLANG TRUE)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "clang")
    set(CMAKE_COMPILER_IS_CLANG TRUE)
  else()
    set(CMAKE_COMPILER_IS_CLANG FALSE)
  endif()
else()
  set(CMAKE_COMPILER_IS_GNUCC_OR_CLANG FALSE)
  set(CMAKE_COMPILER_IS_CLANG FALSE)
endif()

cmake_dependent_option(ENABLE_SSE_AND_AVX_FLAGS "Enable AVX and SSE optimizations (x86-only)" ON "CMAKE_COMPILER_IS_GNUCC_OR_CLANG;IS_X86" OFF)
message("AVX: ${ENABLE_SSE_AND_AVX_FLAGS}")


set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Install prefix" FORCE)
endif()

set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include/Otter"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/Otter"             CACHE PATH "Path where cmake configs will be installed")

set(CMAKE_DEBUG_POSTFIX d)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
if(MSVC AND USE_INTEGRATED_LIBS)
  find_package(PThreads4W REQUIRED)
elseif(MSVC)
  find_package(pthreads REQUIRED)
endif()
find_package(OpenMP)

if(APPLE AND NOT OPENMP_FOUND)
  message(STATUS "  ->  To enable OpenMP on macOS, please install libomp from Homebrew")
endif()

set(ADDITIONAL_CXX_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
set(ADDITIONAL_C_FLAGS "-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors -Wno-deprecated-declarations -Wno-write-strings")
if(UNIX AND BUILD_SHARED_LIBS AND NOT CMAKE_COMPILER_IS_CLANG)
  set(SHAREDLIB_CXX_FLAGS "-Wl,-Bsymbolic")
  set(SHAREDLIB_C_FLAGS "-Wl,-Bsymbolic")
endif()

if(MSVC)
  set(ADDITIONAL_CXX_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  set(ADDITIONAL_C_FLAGS "/wd4013 /wd4018 /wd4028 /wd4047 /wd4068 /wd4090 /wd4101 /wd4113 /wd4133 /wd4190 /wd4244 /wd4267 /wd4305 /wd4477 /wd4996 /wd4819 /fp:fast")
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
endif()

if(CMAKE_COMPILER_IS_GNUCC_OR_CLANG)
  if(CMAKE_COMPILER_IS_CLANG)
    if(UNIX AND NOT APPLE)
      set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")  #force pthread to avoid bugs in some cmake setups
      set(CMAKE_C_FLAGS "-pthread ${CMAKE_C_FLAGS}")
    endif()
  endif()
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  string(REGEX REPLACE "-O0" "-Og" CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})
  if(ENABLE_SSE_AND_AVX_FLAGS)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
  endif()
endif()

set(CMAKE_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} ${SHAREDLIB_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${ADDITIONAL_C_FLAGS} ${SHAREDLIB_C_FLAGS} ${CMAKE_C_FLAGS}")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(FULLPATH_${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/*.hpp")
#add also files in the include folder

#look for all *.c files in src folder
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
#add also .cpp files

if(BUILD_AS_CPP)
  set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
endif()

add_executable(Otter ${sources} ${headers} )

target_include_directories(Otter PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src> $<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}> $<BUILD_INTERFACE:${Stb_INCLUDE_DIR}>)

target_compile_definitions(Otter PRIVATE -DUSE_CMAKE_LIBS)

if(OPENMP_FOUND)
  target_link_libraries(Otter PRIVATE OpenMP::OpenMP_CXX)
  target_link_libraries(Otter PRIVATE OpenMP::OpenMP_C)
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(Otter PRIVATE m)
endif()

if(MSVC)
  target_link_libraries(Otter PRIVATE PThreads_windows::PThreads_windows)
  target_link_libraries(Otter PRIVATE wsock32)
  target_compile_definitions(Otter PRIVATE -D_CRT_RAND_S -DNOMINMAX -D_USE_MATH_DEFINES)
endif()

if(MSVC OR MINGW)
  target_link_libraries(Otter PRIVATE ws2_32)
endif()

target_link_libraries(Otter PRIVATE Threads::Threads)

# Export the package for use from the build-tree (this registers the build-tree with a global CMake-registry)
export(PACKAGE Otter)
