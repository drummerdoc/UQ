# Note: ${TOP} must point to a folder containing BoxLib and Combustion folders
#TOP = ../..
TOP = /scratch1/jbb/

# Put these here until all of them get propagated into the BoxLib make system

HOST=$(shell hostname)
MACHINE=$(shell uname)

ifeq (${HOST},stc-23736s)
  TOP = /Users/rgrout/Code/rev-controlled/LBNL
  WHICHDARWIN=RAY
endif
ifeq ($(findstring Jonathans, $(HOST)), Jonathans)
  WHICHDARWIN=JG
endif

UNAMEN := $(shell uname -n)
ifeq ($(MACHINE), Linux)
  ifeq ($(findstring hedorah, $(UNAMEN)), hedorah)
    WHICHLINUX := HEDORAH
  endif
  ifeq ($(findstring edison, $(UNAMEN)), edison)
    WHICHLINUX := EDISON
  endif
  ifeq ($(findstring mothra, $(UNAMEN)), mothra)
    WHICHLINUX := MOTHRA
  endif
  ifeq ($(findstring gimantis, $(UNAMEN)), gimantis)
    WHICHLINUX := GIMANTIS
  endif
  ifeq ($(findstring peregrine, ${HOSTNAME}), peregrine)
    WHICHLINUX := PEREGRINE
  endif
endif
ifeq (${MACHINE}, Darwin) 
  ifeq ($(findstring MarcsLaptop, $(UNAMEN)), MarcsLaptop)
    WHICHDARWIN:=MARC
  endif
endif

PRECISION       = DOUBLE
DEBUG           = FALSE
#DEBUG           = TRUE
DIM             = 2
#COMP            = CC
#FCOMP           = ftn
COMP            = g++
FCOMP           = gfortran
#USE_MPI         = TRUE
USE_MPI         = FALSE
USE_OMP         = TRUE
#USE_OMP         = FALSE
USE_SDC         = FALSE
USE_SDC         = TRUE
#BUILD_EXEC      = FALSE
BUILD_EXEC      = TRUE
VERBOSE         = TRUE
DO_REGTEST      = FALSE

# Choose model (from list below), and pmf file
#CHEMISTRY_MODEL = DRM19
#CHEMISTRY_MODEL = CHEMH
#CHEMISTRY_MODEL = DAVIS
#CHEMISTRY_MODEL = BURKE
CHEMISTRY_MODEL = BURKEMOD
#NB to us DAVIS chemistry need to hack c. l147 in ChemDriver_F.F
#mcr = Nspec*(19+2*NO+NO*NLITEMAX)+(NO+15)*Nspec**2 * 2                          
#mci = 4*Nspec + NLITEMAX * 2   
# otherwise work arrays for tranlib are too small.
#
#CHEMISTRY_MODEL = LUDME
#CHEMISTRY_MODEL = GRI30

ifeq (${USE_SDC}, TRUE)
  DEFINES += -DLMC_SDC
  #DEFINES += -DDO_AJAC
endif
CFLAGS += -std=c99

#EBASE = main
#EBASE = minimizeEnsemble
#EBASE = evalLocalModel
EBASE = interpSample
#EBASE = EvalExpt
#EBASE = EvalParams

ifeq (${DO_REGTEST}, TRUE)
  include RegTest.mak
endif


ifeq (${BUILD_EXEC}, TRUE)
  USE_SHARED_LIBS := FALSE
  CEXE_sources += ${EBASE}.cpp
else
  USE_SHARED_LIBS := TRUE
endif

# Set paths to BoxLib and Chemistry
BOXLIB_HOME = ${TOP}/BoxLib
CHEMISTRY_DIR = ${TOP}/Combustion/Chemistry
INCLUDE_LOCATIONS += ${CHEMISTRY_DIR}/src_common

include ${BOXLIB_HOME}/Tools/C_mk/Make.defs 
fincludes=${includes}

Bdirs   += ${CHEMISTRY_DIR}/src
Bdirs   += ${BOXLIB_HOME}/Src/C_BaseLib

Blocs    = . 
Bpack	+= $(foreach dir, $(Bdirs), $(dir)/Make.package)
Blocs	+= $(foreach dir, $(Bdirs), $(dir))


include $(Bpack)
INCLUDE_LOCATIONS += $(Blocs)
VPATH_LOCATIONS   += $(Blocs)

CEXE_headers += SimulatedExperiment.H Rand.H ParameterManager.H ExperimentManager.H Driver.H MINPACKstruct.H Minimizer.H Sampler.H UqPlotfile.H
CEXE_sources += SimulatedExperiment.cpp Rand.cpp ParameterManager.cpp ExperimentManager.cpp Driver.cpp Minimizer.cpp Sampler.cpp UqPlotfile.cpp

ifeq (${CHEMISTRY_MODEL}, DAVIS)
  CHEM_MECHFILE = davis.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/gri/PMFs
  VPATH_LOCATIONS += ../extras/Davis05_h2co/chemfiles/
endif
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
ifeq (${CHEMISTRY_MODEL}, BURKEMOD)
  CHEM_MECHFILE = BurkeDryer_mod.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/BurkeDryer_mod
endif
ifeq (${CHEMISTRY_MODEL}, BURKE)
  CHEM_MECHFILE = BurkeDryer.c
  VPATH_LOCATIONS += ${CHEMISTRY_DIR}/data/BurkeDryer
endif
cEXE_sources += ${CHEM_MECHFILE}

ifeq (${USE_SHARED_LIBS}, TRUE)
  # build to link to shared libs
  FFLAGS   += -fPIC
  fFLAGS   += -fPIC
  CFLAGS   += -fPIC
  CXXFLAGS += -fPIC

  ifeq ($(MACHINE), Linux)
    ifeq (${WHICHLINUX}, HEDORAH)
      PYTHON_DIR := /usr
      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
      SHARED_LIBRARY_SUFFIX:=so
      SHARED_LIBRARIES+=-lpython2.7 -lmpi
      PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
      PYTHON_INCLUDES += $(shell python -c 'import mpi4py; print mpi4py.get_include()')
    endif
    ifeq (${WHICHLINUX}, MOTHRA)
      PYTHON_DIR := /usr
      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
      SHARED_LIBRARY_SUFFIX:=so
      SHARED_LIBRARIES+=-lpython2.7 -lmpi
      PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
      PYTHON_INCLUDES += $(shell python -c 'import mpi4py; print mpi4py.get_include()')
#      PYTHON_DIR := /home/marc/src/CCSE/UQ/src/packages/build/pypa-virtualenv-fceb355/pyenv
#      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.6
#      SHARED_LIBRARY_SUFFIX:=so
#      LIBRARIES+=-lpython2.6
    endif
    ifeq (${WHICHLINUX}, GIMANTIS)
      PYTHON_DIR := /home/marc/anaconda/pkgs/python-2.7.7-0
      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
      SHARED_LIBRARY_SUFFIX:=so
      LIBRARIES+=-lpython2.7 -lmpi
      MPILIB := /home/rwgrout/opt/anaconda/lib/libmpich.${SHARED_LIBRARY_SUFFIX}
      PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
      PYTHON_INCLUDES += /home/rwgrout/opt/anaconda/lib/python2.7/site-packages/mpi4py/include
      fFLAGS += -cpp
    endif
    ifeq (${WHICHLINUX}, PEREGRINE)
      PYTHON_DIR := /home/rgrout/opt/anaconda/pkgs/python-2.7.7-0/
      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
      SHARED_LIBRARY_SUFFIX:=so
      LIBRARIES+=-lpython2.7 -lmpi
      MPILIB := /home/rgrout/opt/anaconda/lib/libmpich.${SHARED_LIBRARY_SUFFIX}
      PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
      PYTHON_INCLUDES += $(shell python -c 'import mpi4py; print mpi4py.get_include()')
      fFLAGS += -cpp
    endif
    ifeq (${WHICHLINUX}, EDISON)
      PYTHON_INCLUDES := ${PYTHON_DIR}/include/python2.7
      SHARED_LIBRARY_SUFFIX:=so
      LIBRARIES+=-lpython2.7 -lmpi
      PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
      PYTHON_INCLUDES += $(shell python -c 'import mpi4py; print mpi4py.get_include()')
      fFLAGS += -cpp
    endif
  endif

  ifeq (${MACHINE}, Darwin) 
    SHARED_LIBRARY_SUFFIX:=dylib
    ifeq (${WHICHDARWIN}, RAY)
      #PYTHON_DIR := /opt/local/Library/Frameworks/Python.framework/Versions/Current
      #PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
      PYTHON_DIR := /Users/rgrout/Library/Enthought/Canopy_64bit/User/
      PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
      PYTHON_INCLUDES := /Applications/Canopy.app/appdata/canopy-1.3.0.1715.macosx-x86_64/Canopy.app/Contents/include/python2.7
	  PYTHON_INCLUDES += /Users/rgrout/Library/Enthought/Canopy_64bit/User/lib/python2.7/site-packages/mpi4py/include/
      GFORTLIB := /opt/local/lib/gcc46/libgfortran.${SHARED_LIBRARY_SUFFIX}
      COMP            = g++-mp-4.6
      FCOMP           = gfortran-mp-4.6
	  #COMP = mpicxx
	  #FCOMP = mpif90
      LIBRARIES+=-lpython2.7
      LIBRARIES+= -headerpad_max_install_names
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
    ifeq (${WHICHDARWIN}, JG)
      PYTHON_DIR := /usr/local/Cellar/python/2.7.7_2/Frameworks/Python.framework/Versions/Current
      PYTHON_INCLUDES := ${PYTHON_DIR}/Headers
      GFORTLIB := /usr/local/gfortran/lib/libgfortran.${SHARED_LIBRARY_SUFFIX}
    endif
    SHARED_LIBRARIES += ${PYTHON_DIR}/lib/libpython2.7.${SHARED_LIBRARY_SUFFIX} ${GFORTLIB}
    PYTHON_INCLUDES += $(shell python -c 'import numpy; print numpy.get_include()')
  endif
endif

#
# Where chemkin/premix bits are
VPATH_LOCATIONS += ${CHEMISTRY_DIR}/tools/chemkin/src
INCLUDE_LOCATIONS += ${CHEMISTRY_DIR}/tools/chemkin/src
FEXE_sources += premix_2.F
fEXE_sources += twopnt.f mach.f refine.f cktime.f misc_math.f 
f90EXE_sources += premix_2_util.f90 

PACKAGES=./packages
INCLUDE_LOCATIONS += ${PYTHON_INCLUDES}
INCLUDE_LOCATIONS+=${PACKAGES}/include
INCLUDE_LOCATIONS+=${PACKAGES}/include/cminpack-1

# For static links
LIBRARY_LOCATIONS+=${PACKAGES}/lib
ifeq ($(MACHINE), Linux)
  LIBRARY_LOCATIONS+=${PACKAGES}/lib64
endif
ifeq ($(WHICHLINUX), EDISON)
  LIBRARIES += -dynamic
endif
LIBRARIES += -lcminpack -llapacke -llapack -lblas

# For dynamic links
ifeq ($(MACHINE), Linux)
  SHARED_LIBRARIES+=${PACKAGES}/lib64/libcminpack.${SHARED_LIBRARY_SUFFIX}
  SHARED_LIBRARIES+=${MPILIB}
else
  SHARED_LIBRARIES+=${PACKAGES}/lib/libcminpack.${SHARED_LIBRARY_SUFFIX}
endif
SHARED_LIBRARIES+=${PACKAGES}/lib/liblapacke.${SHARED_LIBRARY_SUFFIX}
SHARED_LIBRARIES+=${PACKAGES}/lib/liblapack.${SHARED_LIBRARY_SUFFIX}
ifeq (${WHICHDARWIN}, RAY)
  SHARED_LIBRARIES+= -headerpad_max_install_names
  LIBRARIES+= -headerpad_max_install_names
endif

PACKAGE := pyemcee

vpath %.c   $(VPATH_LOCATIONS)
vpath %.cpp $(VPATH_LOCATIONS)
vpath %.h   $(VPATH_LOCATIONS)
vpath %.H   $(VPATH_LOCATIONS)
vpath %.F   $(VPATH_LOCATIONS)
vpath %.f   $(VPATH_LOCATIONS)
vpath %.f90 $(VPATH_LOCATIONS)

ifeq ($(VERBOSE), TRUE)
  SUCCESS_MSG := SUCCESS
endif

SWIGDEFS := 
ifeq (${USE_MPI}, TRUE)
  SWIGDEFS+=-DBL_USE_MPI
endif
ifeq (${BUILD_EXEC}, TRUE)
  DEFAULT_TARGET := exec
else
  DEFAULT_TARGET := shared_lib
endif

all: ${DEFAULT_TARGET}

${PACKAGE}/_${PACKAGE}.${SHARED_LIBRARY_SUFFIX}: $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o
	mkdir -p ${PACKAGE}
	$(COMP) -fPIC -shared -o ${PACKAGE}/_${PACKAGE}.${SHARED_LIBRARY_SUFFIX} $(objForExecs) $(objEXETempDir)/${PACKAGE}_wrap.o ${SHARED_LIBRARIES}
	./fixDYLIB.sh ${PACKAGE}/_${PACKAGE}.${SHARED_LIBRARY_SUFFIX}
	cp ${PACKAGE}.py ${PACKAGE}

${PACKAGE}_wrap.cpp: ${PACKAGE}.i
	swig -python -c++ ${SWIGDEFS} -I. $(includes) -I$(SWIG_INCLUDES) -o ${PACKAGE}_wrap.cpp ${PACKAGE}.i

exec: ${executable}
	./fixDYLIBexe.sh ${executable}
	@echo ${SUCCESS_MSG}

shared_lib: ${PACKAGE}/_${PACKAGE}.${SHARED_LIBRARY_SUFFIX}

realclean::
	rm -rf dat dat1 dat2 dat3 ${PACKAGE}/_${PACKAGE}.${SHARED_LIBRARY_SUFFIX} ${PACKAGE}_wrap.cpp ${PACKAGE}.py *.pyc d f o

include $(BOXLIB_HOME)/Tools/C_mk/Make.rules
