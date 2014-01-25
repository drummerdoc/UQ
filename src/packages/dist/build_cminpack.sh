#!/bin/bash

PKG=cminpack
VERS=1.3.0
export CC=gcc
export CFLAGS='-O2'
export CXX='g++'
export CXXFLAGS='-O2'

fullname=$PKG-${VERS}
tardir=$PWD
builddir=$PWD/$fullname-build.$$
destdir=$PWD/../

mkdir "$builddir"
tar -C "$builddir" -xf "$tardir"/$fullname.tar.gz

cd "$builddir/$fullname"
echo hostname > build_$PKG.log
env >> build_$PKG.log

cmake -DCMAKE_INSTALL_PREFIX:STRING="$destdir" -DCMAKE_BUILD_TYPE:STRING="Release" -DSHARED_LIBS:BOOL=TRUE
make -j4
make install
