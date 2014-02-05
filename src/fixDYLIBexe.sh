#!/bin/bash

export MACHINE=`uname`

if [ "${MACHINE}" == "Darwin" ]; then
  for f in $*; do
    install_name_tool -change libcminpack.1.0.90.dylib @loader_path/packages/lib/libcminpack.1.0.90.dylib ${f} 
    install_name_tool -change liblapacke.dylib @loader_path/packages/lib/liblapacke.dylib ${f} 
    install_name_tool -change liblapack.dylib @loader_path/packages/lib/liblapack.dylib ${f} 
  done
fi  
