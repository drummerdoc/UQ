#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <Utility.H>
#include <ChemDriver.H>
#include <cminpack.h>
#include <stdio.h>
#include <lapacke.h>

#include <Driver.H>

#ifdef _OPENMP
#include "omp.h"
#endif

static bool made_cd = false;
ChemDriver *Driver::cd = 0;
MINPACKstruct *Driver::mystruct = 0;
Real Driver::param_eps = 1.e-4;
Real BAD_SAMPLE_FLAG = -1;
Real BAD_DATA_FLAG = -2;

Real 
funcF(void* p, const std::vector<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  s->ResizeWork();

  Real* ptr = const_cast<Real*>(&(pvals[0]));
  ParallelDescriptor::Bcast(ptr,pvals.size(),ParallelDescriptor::IOProcessorNumber());

  // Get prior component of likelihood
  std::pair<bool,Real> Fa = s->parameter_manager.ComputePrior(pvals);

  bool all_and = Fa.first;
  ParallelDescriptor::ReduceBoolAnd(all_and);
  if (all_and != Fa.first) {
      std::cout << "Parameters not compatible across procs" << std::endl;
      BoxLib::Abort();
  }

  if (!Fa.first) {
    return BAD_SAMPLE_FLAG; // Parameter OOB
  }

  // Get data component of likelihood
  std::vector<Real> dvals(s->expt_manager.NumExptData());
  bool ok = s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  if (!ok) {
    return BAD_DATA_FLAG; // Experiment evaluator failed
  }
  Real Fb = s->expt_manager.ComputeLikelihood(dvals);

  // Return sum of pieces
  return (Fa.second  +  Fb);
}

double Driver::LogLikelihood(const std::vector<double>& parameters)
{
  return -funcF((void*)(Driver::mystruct),parameters);
}

int Driver::NumParams()
{
  return Driver::mystruct->parameter_manager.NumParams();
}

int Driver::NumData()
{
  return Driver::mystruct->expt_manager.NumExptData();
}

std::vector<double>
Driver::PriorMean()
{
  return Driver::mystruct->parameter_manager.prior_mean;
}

std::vector<double>
Driver::PriorStd()
{
  return Driver::mystruct->parameter_manager.prior_std;
}

std::vector<double>
Driver::EnsembleStd()
{
  return Driver::mystruct->parameter_manager.ensemble_std;
}

std::vector<double>
Driver::GenerateTestMeasurements(const std::vector<Real>& test_params)
{
  std::vector<Real> test_measurements;
  Driver::mystruct->expt_manager.GenerateTestMeasurements(test_params,test_measurements);
  return test_measurements;
}

std::vector<double>
Driver::LowerBound()
{
  return Driver::mystruct->parameter_manager.lower_bound;
}

std::vector<double>
Driver::UpperBound()
{
  return Driver::mystruct->parameter_manager.upper_bound;
}


/*
 *
 * Constructor for parallel world
 *
 */
//Driver::Driver(int argc, char*argv[], MPI_Comm mpi_comm )
//{
//  BoxLib::Initialize(argc, argv, true, mpi_comm);
//  if (cd == 0) {
//     cd = new ChemDriver;
//     made_cd = true;
//  }
//
//
//  ParmParse pp;
//  param_eps = 1.e-4; pp.query("param_eps",param_eps);
//  mystruct = new MINPACKstruct(*cd,param_eps);
//
//  ParameterManager& parameter_manager = mystruct->parameter_manager;
//  ExperimentManager& expt_manager = mystruct->expt_manager;  
//  expt_manager.InitializeExperiments();
//
//#ifndef DEBUG
//  expt_manager.InitializeTrueData(parameter_manager.TrueParameters());
//  expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)
//#else
//  // For debugging parallel work queue only
//  std::vector<Real> test_measurements, test_params;
//  expt_manager.GenerateTestMeasurements(test_params,test_measurements);
//#endif
//
//}


/*
 *
 * Set mpi communicator to use when init is called
 *
 */
#ifdef BL_USE_MPI
void
Driver::SetComm(MPI_Comm mpi_comm) {
    _mpi_comm = mpi_comm;

}
#endif

/*
 *
 * Constructor for serial world or parallel world where MPI is not initialized yet
 *
 */
Driver::Driver(int argc, char*argv[], int init_later)
{
    // If this is set, expect to do initialization later after mpi world is setup
    if (init_later == 1) {
        _mpi_comm = -1;
        return;
    }
    else {
        init(argc, argv);
    }
}


/*
 *
 * Finish up un-done initialization stuff 
 *
 */
void
Driver::init(int argc, char *argv[])
{

#ifdef BL_USE_MPI
    if (_mpi_comm == -1) {
        MPI_Init (&argc, &argv);
        mpi_initialized = true;
        BoxLib::Initialize(argc, argv, MPI_COMM_WORLD);
    }
    else {
        BoxLib::Initialize(argc, argv, _mpi_comm);
    }
#else
    BoxLib::Initialize(argc, argv);
#endif
    if (cd == 0) {
        cd = new ChemDriver;
        made_cd = true;
    }

    ParmParse pp;
    param_eps = 1.e-4; pp.query("param_eps",param_eps);
    bool use_synthetic_data = false; pp.query("use_synthetic_data",use_synthetic_data);
    if (ParallelDescriptor::IOProcessor() && use_synthetic_data) {
      std::cout << "*************  Using sythetic data " << std::endl;
    }
    mystruct = new MINPACKstruct(*cd,param_eps,use_synthetic_data);

    ParameterManager& parameter_manager = mystruct->parameter_manager;
    ExperimentManager& expt_manager = mystruct->expt_manager;  
    expt_manager.InitializeExperiments();
    expt_manager.InitializeTrueData(parameter_manager.TrueParameters());
    expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)

#if 0
        std::cout << "Running 1 set of experiments" << std::endl;
        std::vector<Real> test_measurements, test_params;
        expt_manager.GenerateTestMeasurements(test_params,test_measurements);
        for (int i=0; i<test_measurements.size(); ++i) {
            std::cout << "Experiment[" << i << "] = " << test_measurements[i] << std::endl;
        }
        std::cout << "Done running 1 set of experiments" << std::endl;
        BoxLib::Abort("Done single test");
#endif
}

Driver::~Driver()
{
  delete mystruct;
  if (made_cd) delete cd;
  BoxLib::Finalize();
#ifdef BL_USE_MPI
  if (mpi_initialized) {
    MPI_Finalize();
  }
#endif
}
