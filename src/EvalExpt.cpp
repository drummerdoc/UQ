#include <Driver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <Utility.H>
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
  bool ioproc = ParallelDescriptor::IOProcessor();


  ParmParse pp;
  std::string sampleFile = "samples"; pp.query("sampleFile",sampleFile);

  bool verbose = false; pp.query("verbose",verbose);
  std::string diagnostic_prefix = "./";
  pp.query("diagnostic_prefix",diagnostic_prefix);

  bool writeExptData = false; pp.query("writeExptData",writeExptData);

#define TESTDATA
#undef TESTDATA

#ifndef TESTDATA
  UqPlotfile pf;
  pf.Read(sampleFile);
  int iter = pf.ITER();
  int iters = pf.NITERS();
  int nwalkers = pf.NWALKERS();
  int ndim = pf.NDIM();
#else
  int iter = 0;
  int iters = 3;
  int nwalkers = 1;
  int ndim = 4;
#endif

  BL_ASSERT(ndim == driver.NumParams());
  std::vector<Real> samples;
  if (ioproc) {
#ifndef TESTDATA
    samples = pf.LoadEnsemble(iter, iters);
#else
    samples.resize(nwalkers*iters*ndim);
    for (int k=0; k<nwalkers; ++k) {
      for (int t=iter; t<iter+iters; ++t) {
	for (int j=0; j<ndim; ++j) {
	  long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	  samples[index] = j;
	}
      }
    }
#endif
  }
  else {
    samples.resize(iters*nwalkers*ndim);
  }
  ParallelDescriptor::ReduceRealSum(&(samples[0]),samples.size());

  std::cout << std::setprecision(8);
  std::cout << std::scientific;
  std::vector<Real> mySamples(ndim);
  std::vector<Real> F(nwalkers*iters,0);

  int nData = driver.NumData();
  std::vector<Real> D;
  if (writeExptData) {
    D.resize(nwalkers*iters*nData,0);
  }
  int nDigits = (int)(std::log10(nwalkers*iters)) + 1;

  std::vector<Real> svals(nData);
  std::vector<Real> dvals(nData);
  Real Fa, Fb;

  for (int k=0; k<nwalkers; ++k) {
    for (int t=iter; t<iter+iters; ++t) {

      int sampleID = nwalkers*(t-iter) + k;

      if (sampleID % nprocs == myproc) {

	for (int j=0; j<ndim; ++j) {
	  long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	  mySamples[j] = samples[index];
	}

	std::string my_prefix = BoxLib::Concatenate(diagnostic_prefix,sampleID,nDigits) + "/";
	expt_manager.SetDiagnosticPrefix(my_prefix);

#ifndef TESTDATA
	F[sampleID] = driver.VerboseLogLikelihood(mySamples,dvals,svals,Fa,Fb);
#else
	F[sampleID] = sampleID;
#endif

	if (writeExptData) {
	  for (int j=0; j<nData; ++j) {
	    long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
#ifndef TESTDATA
	    D[index] = dvals[j];
#else
	    D[index] = j;
#endif
	  }
	}

	if (verbose) {
	  std::cout << sampleID << " [";
	  for (int j=0; j<ndim; ++j) {
	    long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	    std::cout << "  " << samples[index];
	    if (j<ndim-1) std::cout << " ";
	  }
	  std::cout << "] F = " << F[sampleID];
	  if (writeExptData) {
	    std::cout << " D = [";
	    for (int j=0; j<nData; ++j) {
	      long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	      std::cout << "  " << D[index];
	      if (j<nData-1) std::cout << " ";
	    }
	    std::cout << " ]";
	  }
	  std::cout << std::endl;
	}
      }
    }
  }

  ParallelDescriptor::ReduceRealSum(&(F[0]),F.size());
  if (writeExptData) {
    ParallelDescriptor::ReduceRealSum(&(D[0]),D.size());
  }
  if (ioproc) {
    std::string Ffile = "Ffile"; pp.query("Ffile",Ffile);
    UqPlotfile pfi(F,1,1,0,F.size(),"");
    pfi.Write(Ffile);

    if (writeExptData) {
      std::string Dfile = "Dfile"; pp.query("Dfile",Dfile);
      UqPlotfile pfiD(D,nData,nwalkers,iter,iters,"");
      pfiD.Write(Dfile);
    }
  }
  ParallelDescriptor::Barrier();
}

