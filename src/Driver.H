#ifndef _Driver_H_
#define _Driver_H_

#include <ExperimentManager.H>
#include <MINPACKstruct.H>

struct Driver
{
  Driver(int, char **);
  ~Driver();
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();
  static int NumData();
  static std::vector<double> PriorMean();
  static std::vector<double> PriorStd();
  static std::vector<double> GenerateTestMeasurements(const std::vector<Real>& test_params);

  static ChemDriver *cd;
  static MINPACKstruct *mystruct;
  static Real param_eps;
};

#endif
