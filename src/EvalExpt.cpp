#include <Driver.H>
#include <ChemDriver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>

#include <PremixSol.H>
int
main (int   argc,
      char* argv[])
{
//  BoxLib::Initialize(argc,argv); // RG Moved inside driver constructor
  Driver driver(argc,argv);
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  
  const std::vector<Real>& true_data = expt_manager.TrueData();

  int num_data = true_data.size();
  std::cout << "True data: (npts=" << num_data << ")\n"; 
  for(int ii=0; ii<num_data; ii++){
    std::cout << true_data[ii] << std::endl;
  }
}

