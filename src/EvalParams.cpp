#include <Driver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <ParmParse.H>

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& ar) {
  for (int i=0; i<ar.size(); ++i) {
    os << ar[i] << " ";
  }
  return os;
}


int
main (int   argc,
      char* argv[])
{
  Driver driver(argc,argv);
  ParameterManager& param_manager = driver.mystruct->parameter_manager;

  std::cout << param_manager.TrueParameters() << std::endl;

  const std::vector<Real>& active_parameters = param_manager.PriorMean();
  std::cout << active_parameters << std::endl;

  std::cout << "Default values: " << std::endl;
  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    std::cout << "i,v: " << i << ", " << param_manager[i] << std::endl;
  }

  // for (int i=0, End=active_parameters.size(); i<End; ++i) {
  //   param_manager[i] = 7;
  // }

  // std::cout << "New values: " << std::endl;
  // for (int i=0, End=active_parameters.size(); i<End; ++i) {
  //   std::cout << "i,v: " << i << ", " << param_manager[i] << std::endl;
  // }

}

