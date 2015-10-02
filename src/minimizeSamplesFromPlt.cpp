#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>
#include <Utility.H>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <PremixSol.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

int
main (int   argc,
      char* argv[])
{
#ifdef BL_USE_MPI
  MPI_Init (&argc, &argv);
  Driver driver(argc,argv,1);
  driver.SetComm(MPI_COMM_WORLD);
  driver.init(argc,argv);
#else
  Driver driver(argc,argv,0);
#endif

  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();


  ParmParse pp;

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  expt_manager.SetVerbose(false);
  expt_manager.SetParallelMode(ExperimentManager::PARALLELIZE_OVER_THREAD);

  const std::vector<Real>& prior_mean = parameter_manager.PriorMean();
  const std::vector<Real>& ensemble_std = parameter_manager.EnsembleSTD();
  const std::vector<Real>& upper_bound = parameter_manager.UpperBound();
  const std::vector<Real>& lower_bound = parameter_manager.LowerBound();
  int num_params = ensemble_std.size();

  Minimizer* minimizer = 0;
  std::string which_minimizer = "nlls";
  pp.query("which_minimizer",which_minimizer);
  if (which_minimizer == "nlls") {
    minimizer = new NLLSMinimizer();
  }
  else {
    minimizer = new GeneralMinimizer();
  }


  std::string initFile; pp.get("initFile",initFile);
  UqPlotfile pf;
  pf.Read(initFile);
  int iter = pf.ITER();
  int iters = pf.NITERS();
  int nwalkers = pf.NWALKERS();
  int ndim = pf.NDIM();

  pp.query("iter",iter);
  pp.query("iters",iters);
  int walker = 0;
  pp.query("walker",walker);

  std::vector<double> x = pf.LoadEnsemble(iter,iters);

  std::cout << std::setprecision(8);
  std::cout << std::scientific;
  std::vector<Real> samples(ndim);
  std::vector<Real> soln(ndim);

  for (int k=walker; k<nwalkers; ++k) {
    for (int t=iter; t<iter+iters; ++t) {
      for (int j=0; j<ndim; ++j) {
        long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	samples[j] = x[index];
      }

      bool ok = minimizer->minimize((void*)(driver.mystruct), samples, soln);
      std::cout << nwalkers*(t-iter) + k << ": Minpack success? " << ok << std::endl;
    }
  }

  delete minimizer;

  BoxLib::Finalize();

}

