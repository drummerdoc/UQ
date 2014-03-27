#!/bin/bash

export MACHINE=`uname`

if [ "${MACHINE}" == "Darwin" ]; then
  for f in $*; do
    install_name_tool -change libcminpack.1.0.90.dylib @loader_path/packages/lib/libcminpack.1.0.90.dylib ${f} 
    install_name_tool -change liblapacke.dylib @loader_path/packages/lib/liblapacke.dylib ${f} 
    install_name_tool -change liblapack.dylib @loader_path/packages/lib/liblapack.dylib ${f} 
    #install_name_tool -change @rpath/Python /Users/rgrout/Library/Enthought/Canopy_64bit/User/lib/libpython2.7.dylib ./main2d.Darwin.g++-mp-4.6.gfortran.DEBUG.ex
  done
fi  
