#!/usr/bin/env bash

mkdir -p build-ios
pushd build-ios
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=OS -DIOS_ARCH="armv7;arm64;arm64e" \
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
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/ios.toolchain.cmake -DIOS_PLATFORM=SIMULATOR -DIOS_ARCH="i386;x86_64" \
    -DENABLE_BITCODE=1 -DENABLE_ARC=0 -DENABLE_VISIBILITY=0 \
    -DOpenMP_C_FLAGS="-Xclang -fopenmp" -DOpenMP_CXX_FLAGS="-Xclang -fopenmp" \
    -DOpenMP_C_LIB_NAMES="libomp" -DOpenMP_CXX_LIB_NAMES="libomp" \
    -DOpenMP_libomp_LIBRARY="/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/Developer/SDKs/iPhoneSimulator.sdk/usr/lib/libomp.a" \
    -DOTTER_BUILD_BENCHMARK=OFF \
    -DOTTER_MOBILE=ON ..
cmake --build . -j 8
cmake --build . --target install
popd

mkdir -p otter.framework/Versions/A/Headers
mkdir -p otter.framework/Versions/A/Resources
ln -s A otter.framework/Versions/Current
ln -s Versions/Current/Headers otter.framework/Headers
ln -s Versions/Current/Resources otter.framework/Resources
ln -s Versions/Current/otter otter.framework/otter
lipo -create build-ios/install/lib/libotter.a build-ios-sim/install/lib/libotter.a -o otter.framework/Versions/A/otter
cp -r build-ios/install/include/* otter.framework/Versions/A/Headers/
sed -e 's/__NAME__/otter/g' -e 's/__IDENTIFIER__/com.duncan.otter/g' -e 's/__VERSION__/1.0/g' Info.plist > otter.framework/Versions/A/Resources/Info.plist

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
