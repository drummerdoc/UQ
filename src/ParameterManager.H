#ifndef _ParameterManager_
#define _ParameterManager_

#include <ChemDriver.H>

struct ParameterManager
{
  ParameterManager(ChemDriver& _cd) : cd(_cd) {prior_stats_initialized = false;}

  // Add parameter to active set, return default value
  Real AddParameter(int reaction, const ChemDriver::REACTION_PARAMETER& rp);
  void Clear();

  // Reset internal data back to state of initialization
  void ResetParametersToDefault();

  int NumParams() const;

  void SetStatsForPrior(const Array<Real>& mean,
                        const Array<Real>& std);

  void GenerateSampleOfPrior(Array<Real>& parameter_samples) const;

  Real ComputePrior(const Array<Real>& params) const;

  ChemDriver::Parameter& operator[](int i);
  const ChemDriver::Parameter& operator[](int i) const;

  Real TypicalValue(int i);

  bool prior_stats_initialized;
  PArray<ChemDriver::Parameter> active_parameters; // The set of active parameters
  ChemDriver& cd;

  Array<Real> true_prior;
  Array<Real> prior_mean;
  Array<Real> prior_std;
};

inline
ChemDriver::Parameter&
ParameterManager::operator[](int i)
{
  return active_parameters[i];
}

inline
const ChemDriver::Parameter&
ParameterManager::operator[](int i) const
{
  return active_parameters[i];
}

inline
Real
ParameterManager::TypicalValue(int i)
{
  return active_parameters[i].DefaultValue();
}

#endif // ParameterManager
