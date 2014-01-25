
# Note: ${TOP} must point to a folder containing BoxLib and Combustion folders
TOP = ../..
PACKAGES=./packages

PRECISION       = DOUBLE
DEBUG           = TRUE
DEBUG           = FALSE
DIM             = 2
COMP            = g++
FCOMP           = gfortran
USE_MPI         = FALSE
USE_OMP         = TRUE
USE_OMP         = FALSE
USE_SDC         = FALSE
USE_SDC         = TRUE

# Choose model (from list below), and pmf file
#CHEMISTRY_MODEL = DRM19
#CHEMISTRY_MODEL = CHEMH
CHEMISTRY_MODEL = LUDME
#CHEMISTRY_MODEL = GRI30

ifeq (${USE_SDC}, TRUE)
  DEFINES += -DLMC_SDC
  #DEFINES += -DDO_AJAC
endif
CFLAGS += -std=c99

# Set paths to BoxLib and Chemistry
BOXLIB_DIR = ${TOP}/BoxLib
CHEMISTRY_DIR = ${TOP}/Combustion/Chemistry

EBASE = main

include ${BOXLIB_DIR}/Tools/C_mk/Make.defs 
fincludes=${includes}

# build to link to shared libs
FFLAGS   += -fPIC
fFLAGS   += -fPIC
CFLAGS   += -fPIC
CXXFLAGS += -fPIC

Bdirs   += ${CHEMISTRY_DIR}/src
Bdirs   += ${BOXLIB_HOME}/Src/C_BaseLib

Blocs    = . hacked_src
Bpack	+= $(foreach dir, $(Bdirs), $(dir)/Make.package)
Blocs	+= $(foreach dir, $(Bdirs), $(dir))


include $(Bpack)
INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

INCLUDE_LOCATIONS+=${PACKAGES}/include
INCLUDE_LOCATIONS+=${PACKAGES}/include/cminpack-1
LIBRARY_LOCATIONS+=${PACKAGES}/lib
LIBRARY_LOCATIONS+=${PACKAGES}/lib64
LIBRARIES += -lcminpack -llapacke -llapack -lblas
SHARED_LIBRARIES+=${PACKAGES}/lib64/libcminpack.so 
SHARED_LIBRARIES+=${PACKAGES}/lib/liblapacke.so

CEXE_headers += SimulatedExperiment.H Rand.H ParameterManager.H ExperimentManager.H Driver.H MINPACKstruct.H
CEXE_sources += SimulatedExperiment.cpp Rand.cpp ParameterManager.cpp ExperimentManager.cpp Driver.cpp

ifeq (${CHEMISTRY_MODEL}, DRM19)
  CHEM_MECHFILE = drm19.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/gri/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/gri
endif
ifeq (${CHEMISTRY_MODEL}, CHEMH)
  CHEM_MECHFILE = chem-H.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/chem-H/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/chem-H
endif
ifeq (${CHEMISTRY_MODEL}, LIDRY)
  CHEM_MECHFILE = LiDryer.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/LiDryer/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/LiDryer
endif
ifeq (${CHEMISTRY_MODEL}, LIDRYMOD)
  CHEM_MECHFILE = LiDryerMOD.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/LiDryer/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/LiDryer
endif
ifeq (${CHEMISTRY_MODEL}, GRI30_noArN)
  CHEM_MECHFILE = grimech30-noArN.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/gri/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/gri
endif
ifeq (${CHEMISTRY_MODEL}, GLARSKEL)
  CHEM_MECHFILE = glarSkel.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/glar/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/glar
endif
ifeq (${CHEMISTRY_MODEL}, LUDME)
  CHEM_MECHFILE = LuDME.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/Lu/PMFs
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/Lu
endif
cEXE_sources += ${CHEM_MECHFILE}

ifeq ($(MACHINE), Linux)
  ifeq (${WHICHLINUX}, HEDORAH)
    PYTHON_DIR := /usr
    PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.6
    #ifeq ($(USE_MPI), TRUE)
    #  SHARED_LIBRARIES += /usr/local/lib/libmpichcxx.so \
    #                      /usr/local/lib/libmpl.so
    #endif
    #SHARED_LIBRARIES += /usr/lib/libstdc++.so.6 \
    #                    /usr/lib/gcc/x86_64-linux-gnu/4.4.3/libgfortran.so \
    #                    /usr/lib/libm.so
  endif
endif
ifeq (${MACHINE}, Darwin) 
  PYTHON_DIR := /usr/local/Cellar/python/2.7.6/Frameworks/Python.framework/Versions/Current
  PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
  SHARED_LIBRARIES += ${PYTHON_DIR}/lib/libpython2.7.dylib \
                      /usr/local/lib/libgfortran.dylib
  PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
endif

INCLUDE_LOCATIONS += ${PYTHON_INCLUDES}

vpath %.c   $(VPATH_LOCATIONS)
vpath %.cpp $(VPATH_LOCATIONS)
vpath %.h   $(VPATH_LOCATIONS)
vpath %.H   $(VPATH_LOCATIONS)
vpath %.F   $(VPATH_LOCATIONS)
vpath %.f   $(VPATH_LOCATIONS)
vpath %.f90 $(VPATH_LOCATIONS)

PACKAGE := pyemcee

all: _${PACKAGE}.so
#CEXE_sources += main.cpp
#all: $(executable)

_${PACKAGE}.so: all $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o
	g++ -shared -o ${PACKAGE}/_${PACKAGE}.so $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o ${SHARED_LIBRARIES}
	cp ${PACKAGE}.py ${PACKAGE}

${PACKAGE}_wrap.cpp: ${PACKAGE}.i
	swig -python -c++ -I. $(includes) -o ${PACKAGE}_wrap.cpp ${PACKAGE}.i

realclean::
	rm -rf dat dat1 dat2 dat3 _${PACKAGE}.so ${PACKAGE}_wrap.cpp ${PACKAGE}.py *.pyc d f o

include $(BOXLIB_DIR)/Tools/C_mk/Make.rules
