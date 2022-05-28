cmake_minimum_required(VERSION 3.1) # for CMAKE_CXX_STANDARD
set(CMAKE_CXX_STANDARD 11)
macro(otter_install_tool toolname)
    install(TARGETS ${toolname} RUNTIME DESTINATION bin)
endmacro()
if(MSVC)
    # warning C4018: '<': signed/unsigned mismatch
    # warning C4244: 'argument': conversion from 'uint64_t' to 'const unsigned int', possible loss of data
    # warning C4996: 'fopen': This function or variable may be unsafe. Consider using fopen_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.
    add_definitions(/wd4018 /wd4244 /wd4996)
endif()

add_subdirectory(ncnn)
