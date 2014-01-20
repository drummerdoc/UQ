#!/bin/bash

PKG=lapack
VERS=3.5.0
export CC=gcc
export CFLAGS='-O2'
export CXX='g++'
export CXXFLAGS='-O2'
export FC=gfortran

fullname=$PKG-${VERS}
tardir=$PWD
builddir=$PWD/$fullname-build.$$
destdir=$PWD/../

mkdir "$builddir"
tar -C "$builddir" -xf "$tardir"/$fullname.tgz

cd "$builddir/$fullname"
echo hostname > build_$PKG.log
env >> build_$PKG.log

cmake -DCMAKE_INSTALL_PREFIX:STRING="$destdir" -DCMAKE_BUILD_TYPE:STRING="Release"
make -j4
make install
