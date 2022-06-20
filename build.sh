#!/usr/bin/env bash

mkdir -p build-ios
pushd build-ios
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS64 -DIOS_ARCH="arm64;arm64e" \
    -DENABLE_BITCODE=1 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk/usr/lib/libomp.a" \
    -DOTTER_BUILD_BENCHMARK=OFF \
    -DOTTER_MOBILE=ON ..
cmake --build . -j 8
cmake --build . --target install
popd

mkdir -p build-ios-sim
pushd build-ios-sim
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR64 -DIOS_ARCH="x86_64" \
    -DENABLE_BITCODE=1 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib/libomp.a" \
    -DOTTER_BUILD_BENCHMARK=OFF \
    -DOTTER_MOBILE=ON ..
cmake --build . -j 8
cmake --build . --target install
popd

mkdir -p build-mac
pushd build-mac
cmake ..
cmake --build . -j 8
cmake --build . --target install
popd

mkdir -p build-mac-arm
pushd build-mac-arm
cmake -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ..
cmake --build . -j 8
cmake --build . --target install
popd

