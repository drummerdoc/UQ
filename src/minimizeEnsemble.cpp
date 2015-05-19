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
  std::vector<Real> samples(NOS * num_params);
  std::vector<Real> solns(NOS * num_params);

  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();

  std::vector<Real> thisSolns(num_params);
  for (int j=0; j<NOS; ++j)
  {
    bool ok = false;
    std::vector<Real> thisSample;
    while (ok == false) {
      thisSample = GetBoundedSample(prior_mean, ensemble_std, upper_bound, lower_bound);
      ok = minimizer->minimize((void*)(driver.mystruct), thisSample, thisSolns);
    }

    long offset = j * num_params;
    for (int i=0; i<num_params; ++i) {
      samples[offset+i] = thisSample[i];
      solns[offset+i] = thisSolns[i];
    }
  }

  std::vector<Real> samplesg;
  std::vector<Real> solnsg;
  if (myproc == 0) {
    long totalNOS = nprocs * NOS * num_params;
    samplesg.resize(totalNOS);
    solnsg.resize(totalNOS);
  }

  std::cout << "proc " << myproc << "finished" << '\n';

  ParallelDescriptor::Gather(&(samples[0]),NOS*num_params,&(samplesg[0]),0);
  ParallelDescriptor::Gather(&(solns[0]),NOS*num_params,&(solnsg[0]),0);
  if (myproc != 0) {
    samples.clear();
    solns.clear();
  }

  if (myproc == 0) {
    std::string samples_file = "samples"; pp.query("samples",samples_file);
    std::string solns_file = "solns"; pp.query("solns",solns_file);

    UqPlotfile pfi(samplesg,num_params,1,0,NOS*nprocs,"");
    pfi.Write(samples_file);

    UqPlotfile pfo(solnsg,num_params,1,0,NOS*nprocs,"");
    pfo.Write(solns_file);
  }

  delete minimizer;

  BoxLib::Finalize();

}

