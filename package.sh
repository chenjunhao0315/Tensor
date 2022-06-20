#!/usr/bin/env bash

mkdir -p framework
pushd framework
mkdir -p ios
pushd ios
mkdir -p otter.framework/Versions/A/Headers
mkdir -p otter.framework/Versions/A/Resources
ln -s A otter.framework/Versions/Current
ln -s Versions/Current/Headers otter.framework/Headers
ln -s Versions/Current/Resources otter.framework/Resources
ln -s Versions/Current/otter otter.framework/otter
lipo -create ../../build-ios/install/lib/libotter.a ../../build-ios-sim/install/lib/libotter.a -o otter.framework/Versions/A/otter
cp -r ../../build-ios/install/include/* otter.framework/Versions/A/Headers/
sed -e 's/__NAME__/otter/g' -e 's/__IDENTIFIER__/com.duncan.otter/g' -e 's/__VERSION__/1.0/g' Info.plist > otter.framework/Versions/A/Resources/Info.plist
popd
popd

mkdir -p framework
pushd framework
mkdir -p mac
pushd mac
mkdir -p otter.framework/Versions/A/Headers
mkdir -p otter.framework/Versions/A/Resources
ln -s A otter.framework/Versions/Current
ln -s Versions/Current/Headers otter.framework/Headers
ln -s Versions/Current/Resources otter.framework/Resources
ln -s Versions/Current/otter otter.framework/otter
lipo -create ../../build-mac/install/lib/libotter.a -o otter.framework/Versions/A/otter
cp -r ../../build-mac/install/include/* otter.framework/Versions/A/Headers/
sed -e 's/__NAME__/otter/g' -e 's/__IDENTIFIER__/com.duncan.otter/g' -e 's/__VERSION__/1.0/g' Info.plist > otter.framework/Versions/A/Resources/Info.plist
popd
popd

mkdir -p framework
pushd framework
mkdir -p mac-arm
pushd mac-arm
mkdir -p otter.framework/Versions/A/Headers
mkdir -p otter.framework/Versions/A/Resources
ln -s A otter.framework/Versions/Current
ln -s Versions/Current/Headers otter.framework/Headers
ln -s Versions/Current/Resources otter.framework/Resources
ln -s Versions/Current/otter otter.framework/otter
lipo -create ../../build-mac-arm/install/lib/libotter.a -o otter.framework/Versions/A/otter
cp -r ../../build-mac-arm/install/include/* otter.framework/Versions/A/Headers/
sed -e 's/__NAME__/otter/g' -e 's/__IDENTIFIER__/com.duncan.otter/g' -e 's/__VERSION__/1.0/g' Info.plist > otter.framework/Versions/A/Resources/Info.plist
popd
popd
