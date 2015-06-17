#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);
  
  bool ioproc = ParallelDescriptor::IOProcessor();

  ParmParse pp;
  std::string initFile; pp.get("initFile",initFile);
  UqPlotfile pf;
  pf.Read(initFile);
  int iter = pf.ITER();
  int iters = pf.NITERS();
  int nwalkers = pf.NWALKERS();
  int ndim = pf.NDIM();

  std::vector<double> x = pf.LoadEnsemble(iter,iters);

  std::cout << std::setprecision(8);
  std::cout << std::scientific;
  for (int k=0; k<nwalkers; ++k) {
    for (int t=iter; t<iter+iters; ++t) {

      //std::cout << nwalkers*(t-iter) + k << " [";
      for (int j=0; j<ndim; ++j) {
        long index = k + nwalkers*(t-iter) + nwalkers*iters*j;
	std::cout << "  " << x[index];
	if (j<ndim-1) std::cout << " ";
      }
      //std::cout << "]";
      std::cout << std::endl;
    }
  }

  BoxLib::Finalize();
}
