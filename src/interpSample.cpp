#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>

#include <iostream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

int
main (int   argc,
      char* argv[])
{
#ifdef BL_USE_MPI
  MPI_Init (&argc, &argv);
  Driver driver(argc,argv,MPI_COMM_WORLD);
#else
  Driver driver(argc,argv, 0);
#endif

  ParmParse pp;

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  expt_manager.SetVerbose(false);

  std::vector<Real> guess_params;
  const std::vector<Real>& prior_std = parameter_manager.PriorSTD();
  int num_params = prior_std.size();

  int nL = pp.countval("restartL");
  int nR = pp.countval("restartR");

  std::vector<Real> stateL(num_params), stateR(num_params);

  if (nL==1 && nR==1) {

    std::string restartL; pp.get("restartL",restartL);
    std::string restartR; pp.get("restartR",restartR);
    UqPlotfile pf;

    pf.Read(restartL);
    int iter = pf.ITER() + pf.NITERS() - 1;
    int iters = 1;
    int IDL = iter; pp.query("IDL",IDL);
    if (IDL < pf.ITER() || IDL >= iter+iters) {
      BoxLib::Abort("IDL sample not in restartL");
    }
    stateL = pf.LoadEnsemble(IDL, iters);
    BL_ASSERT(stateL.size() == num_params);

    pf.Read(restartR);
    iter = pf.ITER() + pf.NITERS() - 1;
    iters = 1;
    int IDR = iter; pp.query("IDR",IDR);
    if (IDR < pf.ITER() || IDR >= iter+iters) {
      BoxLib::Abort("IDR sample not in restartR");
    }
    stateR = pf.LoadEnsemble(IDR, iters);
    BL_ASSERT(stateR.size() == num_params);
  }
  else {
    int nsL = pp.countval("stateL");
    int nsR = pp.countval("stateR");
    BL_ASSERT(nsL == nsR == num_params);
    pp.getarr("stateL",stateL,0,num_params);
    pp.getarr("stateR",stateR,0,num_params);
  }

  int intervals=10; pp.query("intervals",intervals);
  int niters = intervals + 1;
  std::vector<Real> samples(num_params * niters);
  std::vector<Real> sample(num_params);
  for (int n=0; n<niters; ++n) {
    Real eta = (Real) n / intervals;
    for (int i=0; i<num_params; ++i) {
      sample[i] = (1 - eta)*stateL[i] + eta*stateR[i];
      int index = n + niters*i;
      samples[index] = sample[i];
    }
    Real F = NegativeLogLikelihood(sample);
  }

  BoxLib::Finalize();
}

