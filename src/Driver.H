#ifndef _Driver_H_
#define _Driver_H_

#include <ExperimentManager.H>
#include <MINPACKstruct.H>

struct Driver
{
  Driver();
  ~Driver();
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();

  static ChemDriver *cd;
  static MINPACKstruct *mystruct;
  static Real param_eps;
};

#endif
