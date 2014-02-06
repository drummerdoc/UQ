
# Note: ${TOP} must point to a folder containing BoxLib and Combustion folders
TOP = ../..
PACKAGES=./packages

HOST=$(shell hostname)
ifeq (${HOST},stc-23736s)
TOP = /Users/rgrout/Code/rev-controlled/LBNL
WHICHDARWIN=RAY
endif

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
BUILD_EXEC      = TRUE

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
BOXLIB_HOME = ${TOP}/BoxLib
CHEMISTRY_DIR = ${TOP}/Combustion/Chemistry
INCLUDE_LOCATIONS += ${CHEMISTRY_DIR}/src_common

EBASE = main

include ${BOXLIB_HOME}/Tools/C_mk/Make.defs 
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
    SHARED_LIBRARY_SUFFIX:=so
    LIBRARIES+=-lpython2.6
  endif
endif
ifeq (${MACHINE}, Darwin) 
  SHARED_LIBRARY_SUFFIX:=dylib
  ifeq (${WHICHDARWIN}, RAY)
    PYTHON_DIR := /opt/local/Library/Frameworks/Python.framework/Versions/Current
    PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
    GFORTLIB := /opt/local/lib/gcc46/libgfortran.${SHARED_LIBRARY_SUFFIX}
    COMP            = g++-mp-4.6
    FCOMP           = gfortran-mp-4.6
  endif
  ifeq (${WHICHDARWIN}, MARC)
    PYTHON_DIR := /usr/local/Cellar/python/2.7.6/Frameworks/Python.framework/Versions/Current
    PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
    GFORTLIB := /usr/local/lib/libgfortran.${SHARED_LIBRARY_SUFFIX}
    LIBRARIES+=-lpython2.7
  endif
  ifeq (${WHICHDARWIN}, MATTI)
    PYTHON_DIR := /System/Library/Frameworks/Python.framework/Versions/2.7
    PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
    GFORTLIB := /opt/local/lib/gcc47/libgfortran.${SHARED_LIBRARY_SUFFIX}
    LIBRARIES+=-lpython2.7
  endif
  SHARED_LIBRARIES += ${PYTHON_DIR}/lib/libpython2.7.${SHARED_LIBRARY_SUFFIX} ${GFORTLIB}
  PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
endif

MY_SHARED_LIBRARY_SUFFIX:=so
INCLUDE_LOCATIONS += ${PYTHON_INCLUDES}

INCLUDE_LOCATIONS+=${PACKAGES}/include
INCLUDE_LOCATIONS+=${PACKAGES}/include/cminpack-1

# For static links
LIBRARY_LOCATIONS+=${PACKAGES}/lib
#LIBRARY_LOCATIONS+=${PACKAGES}/lib64
LIBRARIES += -lcminpack -llapacke -llapack -lblas

# For dynamic links
SHARED_LIBRARIES+=${PACKAGES}/lib/libcminpack.${SHARED_LIBRARY_SUFFIX}
SHARED_LIBRARIES+=${PACKAGES}/lib/liblapacke.${SHARED_LIBRARY_SUFFIX}
SHARED_LIBRARIES+=${PACKAGES}/lib/liblapack.${SHARED_LIBRARY_SUFFIX}

PACKAGE := pyemcee

vpath %.c   $(VPATH_LOCATIONS)
vpath %.cpp $(VPATH_LOCATIONS)
vpath %.h   $(VPATH_LOCATIONS)
vpath %.H   $(VPATH_LOCATIONS)
vpath %.F   $(VPATH_LOCATIONS)
vpath %.f   $(VPATH_LOCATIONS)
vpath %.f90 $(VPATH_LOCATIONS)

EBASE=main

ifeq ($(BUILD_EXEC), TRUE)
  test: exec
endif

all: ${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX}

${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX}: $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o
	mkdir -p ${PACKAGE}
	$(COMP) -shared -o ${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX} $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o ${SHARED_LIBRARIES}
	./fixDYLIB.sh ${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX}
	cp ${PACKAGE}.py ${PACKAGE}

${PACKAGE}_wrap.cpp: ${PACKAGE}.i
	swig -python -c++ -I. $(includes) -o ${PACKAGE}_wrap.cpp ${PACKAGE}.i

exec: main.o ${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX}
	@echo Linking $@ ...
	$(SILENT) $(PRELINK) $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) \
		-o  $(addsuffix $(optionsSuffix).ex, $(EBASE)) $^ ${SHARED_LIBRARIES} ${libraries}
	./fixDYLIBexe.sh $(addsuffix $(optionsSuffix).ex, $(EBASE))

realclean::
	rm -rf dat dat1 dat2 dat3 ${PACKAGE}/_${PACKAGE}.${MY_SHARED_LIBRARY_SUFFIX} ${PACKAGE}_wrap.cpp ${PACKAGE}.py *.pyc d f o

include $(BOXLIB_HOME)/Tools/C_mk/Make.rules
