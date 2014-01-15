
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
USE_OMP         = FALSE
USE_OMP         = TRUE
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

EBASE = driver
#EBASE = findInterestingParams
#EBASE = Minimizex2

Blocs := . hacked_src
INCLUDE_LOCATIONS += hacked_src

INCLUDE_LOCATIONS+=${PACKAGES}/include
INCLUDE_LOCATIONS+=${PACKAGES}/include/cminpack-1
LIBRARY_LOCATIONS+=${PACKAGES}/lib
LIBRARY_LOCATIONS+=${PACKAGES}/lib64
LIBRARIES += -lcminpack -llapacke -llapack -lblas

CEXE_headers += SimulatedExperiment.H Rand.H ParameterManager.H ExperimentManager.H
CEXE_sources += ${EBASE}.cpp SimulatedExperiment.cpp Rand.cpp ParameterManager.cpp ExperimentManager.cpp

include $(CHEMISTRY_DIR)/tools/make/Make.CHEM
