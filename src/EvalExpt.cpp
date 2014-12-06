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

  std::vector<Real> myparams = param_manager.TrueParameters();
  int nParams = myparams.size();
  ParmParse pp;

  Array<std::string> pFabs;
  int num_pfabs = pp.countval("pFabs");
  if (pp.countval("pvals")==nParams) {
    pp.getarr("pvals",myparams,0,nParams);
  }
  else if (num_pfabs > 0) {
    pp.getarr("pFabs",pFabs,0,num_pfabs);
  }

  int num_datasets = (num_pfabs > 0 ? num_pfabs : 1);
  int num_data = true_data.size();
  std::vector<Real> data(num_data);

  for (int i=0; i<num_datasets; ++i) {

    if (num_pfabs > 0) {
      if (ParallelDescriptor::IOProcessor()) {
        std::cout << "...Reading parameters from " << pFabs[i] << std::endl;
      }
      std::ifstream ifs; ifs.open(pFabs[i].c_str());
      FArrayBox pfab;    pfab.readFrom(ifs); ifs.close();
      BL_ASSERT(pfab.nComp()==nParams);
      pfab.getVal(&(myparams[0]),pfab.smallEnd(),0,nParams);
    }

    bool ok = expt_manager.GenerateTestMeasurements(myparams,data);
  
    if (ParallelDescriptor::IOProcessor()) {

      if (!ok) {
        //BoxLib::Abort("Measurements bad");
      }
      else {
        for(int ii=0; ii<num_data; ii++){
          std::cout << ii << " (" << expt_manager.ExperimentNames()[ii] << ") "
                    << '\t' << true_data[ii] << '\t' << data[ii] << std::endl;
        }
      }

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
}

