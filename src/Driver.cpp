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
Real F_UNSET_FLAG = -100;

const std::vector<Real>&
Driver::MeasuredDataSTD()
{
  return Driver::mystruct->expt_manager.ObservationSTD();
}

const std::vector<Real>&
Driver::MeasuredData()
{
  return Driver::mystruct->expt_manager.TrueDataWithObservationNoise();
}

Real 
funcF(void* p, const std::vector<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  s->ResizeWork();
  std::vector<Real> dvals(s->expt_manager.NumExptData());

  // Get prior component of likelihood
  std::pair<bool,Real> Fa = s->parameter_manager.ComputePrior(pvals);

  Real F = F_UNSET_FLAG;
  if (!Fa.first) {

    F = BAD_SAMPLE_FLAG; // Parameter OOB

  } else {

    // Get data component of likelihood
    bool ok = s->expt_manager.GenerateTestMeasurements(pvals,dvals);
    if (!ok) {

      F = BAD_DATA_FLAG; // Experiment evaluator failed

    } else {

      Real Fb = s->expt_manager.ComputeLikelihood(dvals);
      F = Fa.second + Fb;

    }
  }

#if 0
  std::cout << "X = { ";
  for(int i=0; i<pvals.size(); i++){
    std::cout << pvals[i] << " ";
  }
  std::cout << "}, D = { ";
  for(int i=0; i<s->expt_manager.NumExptData(); i++){
    std::cout << dvals[i] << " ";
  }
  std::cout << "}, F = " << F << std::endl;
#endif

#if 0
  std::cout << "X = { ";
  for(int i=0; i<pvals.size(); i++){
    std::cout << pvals[i] << " ";
  }
  std::cout << "}, D = { ";
  for(int i=0; i<s->expt_manager.NumExptData(); i++){
    std::cout << dvals[i] << " ";
  }

  const std::vector<Real>& data = s->expt_manager.TrueDataWithObservationNoise();
  const std::vector<Real>& obs_std = s->expt_manager.ObservationSTD();

  std::cout << "}, S = { ";
  for(int i=0; i<s->expt_manager.NumExptData(); i++){
    std::cout << (data[i] - dvals[i])/obs_std[i] << " ";
  }
  std::cout << "}, F = " << F << std::endl;
#endif

  return F;
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
Driver::TrueParameters()
{
  return Driver::mystruct->parameter_manager.TrueParameters();
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
    omp_threads_override = -1;
    if (init_later == 1) {
        _mpi_comm = MPI_COMM_NULL;
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
    if (_mpi_comm == MPI_COMM_NULL) {
      MPI_Init (&argc, &argv);
      mpi_initialized = true;
      BoxLib::Initialize(argc, argv, MPI_COMM_WORLD);
    }
    else {
      mpi_initialized = false;
      BoxLib::Initialize(argc, argv, _mpi_comm);
    }
#else
    BoxLib::Initialize(argc, argv);
#endif
#ifdef BL_USE_OMP
   if(omp_threads_override > 0){
      omp_set_num_threads(omp_threads_override);
      std::cout << "Setting thread count to " << omp_threads_override << std::endl;
   }
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
    if (ParallelDescriptor::IOProcessor() ) {
      std::cout << "=================== REACTION MECHANISM =================" << std::endl;
      parameter_manager.PrintActiveParams();
      std::cout << "========================================================" << std::endl;
      
    }
    ExperimentManager& expt_manager = mystruct->expt_manager;  
    expt_manager.InitializeExperiments();
    expt_manager.InitializeTrueData(parameter_manager.TrueParameters());
    expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)

#if 0
        std::cout << "Running 1 set of experiments" << std::endl;

        std::vector<Real> test_measurements, test_params;

        const std::vector<Real>& prior_std = parameter_manager.PriorSTD();
        int num_params = prior_std.size();
        std::vector<Real> stateL(num_params);
        int nsL = pp.countval("stateL");
        BL_ASSERT(nsL == num_params);
        pp.getarr("stateL",stateL,0,num_params);

        test_params.resize(num_params);
        test_params = stateL;

        const std::vector<Real>& true_params = parameter_manager.TrueParameters();

        for (int i=0; i<test_params.size(); ++i) {
            std::cout << "Parameter[" << i << "] = " << test_params[i] << "; True = " << true_params[i] << std::endl;
        }

        expt_manager.GenerateTestMeasurements(test_params,test_measurements);
        for (int i=0; i<test_measurements.size(); ++i) {
            std::cout << "Experiment[" << i << "] = " << test_measurements[i] << std::endl;
        }
        std::cout << "Done running 1 set of experiments" << std::endl;
        BoxLib::Abort("Done single test");
#endif
}

void
Driver::SetParallelModeThreaded()
{
    mystruct->expt_manager.SetVerbose(false);
    mystruct->expt_manager.SetParallelMode(ExperimentManager::PARALLELIZE_OVER_THREAD);
}

void
Driver::SetNumThreads(int num_threads)
{
    omp_threads_override = num_threads;
}


Driver::~Driver()
{
  delete mystruct;
  if (made_cd) delete cd;
  if( mpi_initialized ){
	  BoxLib::Finalize(mpi_initialized);
  }
}
