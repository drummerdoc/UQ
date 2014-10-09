#include <Driver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <ParmParse.H>

int
main (int   argc,
      char* argv[])
{
//  BoxLib::Initialize(argc,argv); // RG Moved inside driver constructor
  bool use_synthetic_data = false;
  Driver driver(argc,argv,use_synthetic_data);
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  ParameterManager& param_manager = driver.mystruct->parameter_manager;
  
  const std::vector<Real>& true_data = expt_manager.TrueData();

  int num_data = true_data.size();
  std::vector<Real> data(num_data);
  if (ParallelDescriptor::IOProcessor()) {
    std::cout << "Computing targets...:\n"; 
  }
  expt_manager.GenerateTestMeasurements(param_manager.TrueParameters(),data);
  if (ParallelDescriptor::IOProcessor()) {
    for(int ii=0; ii<num_data; ii++){
      std::cout << ii << " (" << expt_manager.ExperimentNames()[ii] << ") "
                << '\t' << true_data[ii] << '\t' << data[ii] << std::endl;
    }

    ParmParse pp;
    if (pp.countval("outfile") > 0) {
      std::string outfile; pp.get("outfile",outfile);
      Box box(IntVect(D_DECL(0,0,0)),
              IntVect(D_DECL(true_data.size()-1,0,0)));
      FArrayBox outfab(box,1);
      for(int ii=0; ii<num_data; ii++){
        IntVect iv(D_DECL(ii,0,0));
        outfab(iv,0) = true_data[ii];
      }
      std::cout << "Writing data to " << outfile << std::endl;
      std::ofstream ofs;
      ofs.open(outfile.c_str());
      outfab.writeOn(ofs);
      ofs.close();
    }
  }
}

