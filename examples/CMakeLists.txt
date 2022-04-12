macro(otter_add_example name)
    add_executable(${name} ${name}.cpp)

    target_link_libraries(${name} PRIVATE otter)

    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Tensor)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Tensor)

otter_add_example(Nanodet)
otter_add_example(YoloFastest)
