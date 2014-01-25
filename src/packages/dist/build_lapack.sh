#!/bin/bash

PKG=lapack
VERS=3.5.0
#export CC='gcc-mp-4.6'
#export CXX='g++-mp-4.6'
#export FC=gfortran-mp-4.6

export CC='gcc'
export CXX='g++'
export FC=gfortran

export CFLAGS='-O2'
export CXXFLAGS='-O2'

fullname=$PKG-${VERS}
tardir=$PWD
builddir=$PWD/$fullname-build.$$
destdir=$PWD/../

mkdir "$builddir"
tar -C "$builddir" -xf "$tardir"/$fullname.tgz

cd "$builddir/$fullname"
echo hostname > build_$PKG.log
env >> build_$PKG.log

cmake -DCMAKE_INSTALL_PREFIX:STRING="$destdir" -DCMAKE_BUILD_TYPE:STRING="Release" -DBUILD_SHARED_LIBS:BOOL=TRUE
make -j4
make install
