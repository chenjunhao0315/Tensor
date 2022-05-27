if(MSVC)
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4996)
endif()

add_executable(benchotter benchotter.cpp)
target_link_libraries(benchotter PRIVATE otter)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(benchotter PRIVATE nodefs.js)
endif()

# add benchncnn to a virtual project group
set_property(TARGET benchotter PROPERTY FOLDER "benchmark")
