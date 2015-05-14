#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>
#include <Utility.H>

#include <iostream>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <PremixSol.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

std::vector<Real>
GetBoundedSample(const std::vector<Real>& prior_mean,
		 const std::vector<Real>& prior_std,
		 const std::vector<Real>& upper_bound,
		 const std::vector<Real>& lower_bound)
{
  int N = prior_mean.size();
  std::vector<Real> sample(N);

  for (int i=0; i<N; ++i) {
    sample[i] = prior_mean[i] + prior_std[i]*BoxLib::Random();
    bool sample_oob = (sample[i] < lower_bound[i] || sample[i] > upper_bound[i]);	
    while (sample_oob) {
      sample[i] = prior_mean[i] + prior_std[i]*BoxLib::Random();
      sample_oob = (sample[i] < lower_bound[i] || sample[i] > upper_bound[i]);
    }
  }
  return sample;
}

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

  int NOS = 1; pp.query("NOS",NOS);
  std::vector<std::vector<Real> > samples(NOS, std::vector<Real>(num_params,-1));
  std::vector<std::vector<Real> > solns(NOS, std::vector<Real>(num_params,-1));

  for (int j=0; j<NOS; ++j)
  {
    samples[j] = GetBoundedSample(prior_mean, ensemble_std, upper_bound, lower_bound);

    //Real F = NegativeLogLikelihood(samples[j]);
    minimizer->minimize((void*)(driver.mystruct), samples[j], solns[j]);
  }

  delete minimizer;

  BoxLib::Finalize();

}

