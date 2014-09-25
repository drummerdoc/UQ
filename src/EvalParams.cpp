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


std::ostream& operator<<(std::ostream& os, ReactionData* r) {
  os << "Fwd: (" << r->fwd_A << " " << r->fwd_beta << " " << r->fwd_Ea << ") ";
  if (r->is_PD) {
    os << "Low: (" << r->low_A << " " << r->low_beta << " " << r->low_Ea << ") ";
    os << "Troe: (" << r->troe_a  << " " << r->troe_Ts  << " " << r->troe_Tss << " " << r->troe_Tsss << ") ";
    os << "Sri: (" << r->sri_a << " " << r->sri_b << " " << r->sri_c << " " << r->sri_d << " " << r->sri_e << ") ";
  }
  if (r->rev_A != 0  &&  r->rev_beta != 0  &&  r->rev_Ea != 0) {
    os << "Rev (" << r->rev_A  << " " << r->rev_beta  << " " << r->rev_Ea << ") ";
  }
  os << "Units: (" << r->activation_units << " " << r->prefactor_units << " " << r->phase_units << ") ";
  if (r->nTB > 0) {
    os << "TB: (";
    for (int j=0; j<r->nTB; ++j) {
      os << " (" << r->TBid[j] << ":" << r->TB[j] << ")";
    }
    os << ") ";
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

  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    param_manager[i] = 7;
  }

  std::cout << "New values: " << std::endl;
  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    std::cout << "i,v: " << i << ", " << param_manager[i] << std::endl;
  }

  std::ofstream os; os.open("pvals1.dat");
  for (int i=0; i<driver.cd->numReactions(); ++i) {
    struct ReactionData* r = get_reaction_parameters(i);
    os << i << ": " << r << std::endl;
  }
  os.close();

  os.open("pvals1D.dat");
  for (int i=0; i<driver.cd->numReactions(); ++i) {
    struct ReactionData* r = get_default_reaction_parameters(i);
    os << i << ": " << r << std::endl;
  }
  os.close();

  param_manager.ResetParametersToDefault();

  std::cout << "Reset to default values: " << std::endl;
  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    std::cout << "i,v: " << i << ", " << param_manager[i] << std::endl;
  }

}

