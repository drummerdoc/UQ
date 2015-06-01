#include <Driver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <ParmParse.H>
#include <UqPlotfile.H>

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

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  expt_manager.SetVerbose(false);
  expt_manager.SetParallelMode(ExperimentManager::PARALLELIZE_OVER_THREAD);

  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();


  ParmParse pp;
  std::string sampleFile = "samples"; pp.query("sampleFile",sampleFile);

  bool verbose = false; pp.query("verbose",verbose);

  UqPlotfile pf;
  pf.Read(sampleFile);
  int iter = pf.ITER();
  int iters = pf.NITERS();
  int nwalkers = pf.NWALKERS();
  int ndim = pf.NDIM();
  BL_ASSERT(ndim == driver.NumParams());
  std::vector<Real> samples = pf.LoadEnsemble(iter, iters);

  std::cout << std::setprecision(8);
  std::cout << std::scientific;
  std::vector<Real> mySamples(ndim);
  for (int k=0; k<nwalkers; ++k) {
    for (int t=iter; t<iter+iters; ++t) {

      int sampleID = nwalkers*(t-iter) + k;

      for (int j=0; j<ndim; ++j) {
	long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	mySamples[j] = samples[index];
      }

      if (verbose) {
	std::cout << sampleID << " [";
	for (int j=0; j<ndim; ++j) {
	  long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	  std::cout << "  " << samples[index];
	  if (j<ndim-1) std::cout << " ";
	}
	std::cout << "]" << std::endl;
      }
      double myF = driver.LogLikelihood(mySamples);
    }
  }

}

