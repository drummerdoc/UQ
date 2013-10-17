#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <Utility.H>
#include <ChemDriver.H>
#include <cminpack.h>

#include <ExperimentManager.H>

static
void 
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " pmf_file=<input fab file name> [options] \n";
  std::cerr << "\tOptions:       Patm = <pressure, in atmospheres> \n";
  std::cerr << "\t                 dt = <time interval, in seconds> \n";
  std::cerr << "\t     time_intervals = <number of intervals to collect data> \n";
  std::cerr << "\t              Tfile = <T search value, in K> \n";
  exit(1);
}


// A simple struct to hold all the parameters, experiment data and some work space
// to be accessible to the MINPACK function
struct MINPACKstruct
{
  MINPACKstruct(ChemDriver& cd, Real _param_eps)
  : parameter_manager(cd), expt_manager(parameter_manager), param_eps(_param_eps) {}

  void ResizeWork() {
    int N = parameter_manager.NumParams();
    int NWORK = 2;
    work.resize(NWORK);
    for (int i=0; i<NWORK; ++i) {
      work[i].resize(N);
    }
  }

  ParameterManager parameter_manager;
  ExperimentManager expt_manager;
  Array<Array<Real> > work;
  Real param_eps;
};

static
Real
get_macheps() {
  Real h = 1;
  while (1+h != 1) {
    h *= 0.5;
  }
  Real mach_eps = std::sqrt(h);
  std::cout << "Setting macheps to " << mach_eps << std::endl;
  return mach_eps;
}

static ChemDriver* cd;
MINPACKstruct *mystruct;


int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  if (argc<2) print_usage(argc,argv);

  cd = new ChemDriver;

  Real eps = get_macheps();
  Real param_eps = 1000*eps;
  mystruct = new MINPACKstruct(*cd,param_eps);

  ParameterManager& parameter_manager = mystruct->parameter_manager;
  ExperimentManager& expt_manager = mystruct->expt_manager;  
  
  CVReactor cv_reactor(*cd);
  expt_manager.AddExperiment(cv_reactor,"exp1");
  expt_manager.InitializeExperiments();

  for (int j=0; j<cd->numReactions(); ++j) {
    parameter_manager.Clear();
    Array<Real> true_params;
    // Reactions that seem to matter: 0, 15, 41, 49, 135 (15, 135 strongest)
    true_params.push_back(parameter_manager.AddParameter(j,ChemDriver::FWD_EA));
    int num_params = parameter_manager.NumParams();

    Array<Real> prior_mean(num_params);
    for(int ii=0; ii<num_params; ii++){
      prior_mean[ii] = true_params[ii] * 1.5;
      if (prior_mean[ii] == 0) {prior_mean[ii] =1e-2;}
    }

    int num_data = expt_manager.NumExptData();
    Array<Real> prior_data(num_data);
    expt_manager.GenerateTestMeasurements(prior_mean,prior_data);
    
    for(int ii=num_data-1; ii<num_data; ii++){
      std::cout << j << " " << prior_data[ii] << std::endl;
    }    
  }
  delete mystruct;
  delete cd;

  BoxLib::Finalize();
}

