cmake_minimum_required(VERSION 3.4)

project(pyotter)

set(PACKAGE_VERSION ${OTTER_VERSION_STRING})
add_definitions(-DVERSION_INFO="${PACKAGE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(pybind11)

if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    if(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/otter/)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/otter/)
    endif(MSVC OR CMAKE_GENERATOR STREQUAL "Xcode")
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

include_directories(${pybind11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
pybind11_add_module(pyotter src/main.cpp)
set_target_properties(pyotter PROPERTIES OUTPUT_NAME "otter")
target_link_libraries(pyotter PUBLIC otter)
set_target_properties(pyotter PROPERTIES PREFIX "" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/otter")
set_property(TARGET pyotter PROPERTY FOLDER "python")
if("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")
    add_custom_command(TARGET pyotter POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/otter/otter${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION} 
        ${PROJECT_SOURCE_DIR}/otter/otter${PYTHON_MODULE_PREFIX}${PYTHON_MODULE_EXTENSION})
endif("${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" STREQUAL "")

configure_file(setup.py.i ${PROJECT_SOURCE_DIR}/setup.py)
