#!/bin/bash

export PKG=cminpack
export VERS=1.3.0

export BUILD_TYPE="Release"

export HOST=`hostname`
if [ "${HOST}" == "stc-23736s" ]; then
  export CC='gcc-mp-4.6'
fi

export fullname=${PKG}-${VERS}
tardir=${PWD}
builddir=${PWD}/${fullname}-${BUILD_TYPE}
destdir=${PWD}/..

mkdir -p "${builddir}"
tar -C "${builddir}" -xzf "${tardir}"/${fullname}.tar.gz

cd "${builddir}/${fullname}"

if [[ "${HOST}" == "edison"* ]]; then
  cmake -DCMAKE_INSTALL_PREFIX:STRING="${destdir}" \
        -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
        -DSHARED_LIBS:BOOL=TRUE \
        -DCMAKE_SHARED_LINKER_FLAGS="-dynamic"\
        -DCMAKE_EXE_LINKER_FLAGS="-dynamic"
else
  cmake -DCMAKE_INSTALL_PREFIX:STRING="${destdir}" \
        -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
        -DSHARED_LIBS:BOOL=TRUE 
fi

make -j4

make install

