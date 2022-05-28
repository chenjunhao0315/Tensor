#look for all *.h files in src folder
file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp")
#add also files in the include folder

#look for all *.c files in src folder
file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
#add also .cpp files

add_executable(ncnn2otter ${sources} ${headers} )

set_property(TARGET ncnn2otter PROPERTY FOLDER "tools/converter")
otter_install_tool(ncnn2otter)
