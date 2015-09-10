#!/bin/bash

export PKG=lapack
export VERS=3.5.0

export BUILD_TYPE="Release"

export fullname=${PKG}-${VERS}
export tardir=${PWD}
export builddir=${PWD}/${fullname}-${BUILD_TYPE}
export destdir=${PWD}/..

mkdir -p "${builddir}"
tar -C "${builddir}" -xzf "${tardir}"/${fullname}.tgz
cd "${builddir}/${fullname}"

export HOST=`hostname`
if [ "${HOST}" == "stc-23736s" ]; then
  export CC='gcc-mp-4.6'
  export FC='gfortran-mp-4.6'
fi
if [ "${HOST}" == "gimantis" ]; then
  export CC='gcc'
  export FC='gfortran'
fi


# edison needs flag for dynamic libraries
if [[ "${HOST}" == "edison"* ]]; then
  cmake -DCMAKE_INSTALL_PREFIX:STRING="${destdir}" \
        -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
        -DBUILD_SHARED_LIBS:BOOL=TRUE \
        -DLAPACKE:BOOL=TRUE \
        -DCMAKE_SHARED_LINKER_FLAGS="-dynamic"
else
  cmake -DCMAKE_INSTALL_PREFIX:STRING="${destdir}" \
        -DCMAKE_BUILD_TYPE:STRING=${BUILD_TYPE} \
        -DBUILD_SHARED_LIBS:BOOL=TRUE \
        -DLAPACKE:BOOL=TRUE
fi

make -j4

make install

export MACHINE=`uname`

if [ "${MACHINE}" == "Darwin" ]; then
  install_name_tool -change liblapack.dylib @loader_path/liblapack.dylib ${destdir}/lib/liblapacke.dylib 
  install_name_tool -change libblas.dylib @loader_path/libblas.dylib ${destdir}/lib/liblapacke.dylib 
fi  
