#ifndef _ParameterManager_
#define _ParameterManager_

#include <ChemDriver.H>

struct ParameterManager
{
  ParameterManager(ChemDriver& _cd);

  // Add parameter to active set, return default value
  Real AddParameter(int reaction, const ChemDriver::REACTION_PARAMETER& rp);
  void Clear();

  // Reset internal data back to state of initialization
  void ResetParametersToDefault();

  int NumParams() const;

  void SetStatsForPrior(const std::vector<Real>& mean,
                        const std::vector<Real>& std,
                        const std::vector<Real>& lower_bound,
                        const std::vector<Real>& upper_bound);

  void GenerateSampleOfPrior(std::vector<Real>& parameter_samples) const;

  std::pair<bool,Real> ComputePrior(const std::vector<Real>& params) const;

  ChemDriver::Parameter& operator[](int i);
  const ChemDriver::Parameter& operator[](int i) const;

  Real TypicalValue(int i);

  bool prior_stats_initialized;
  PArray<ChemDriver::Parameter> active_parameters; // The set of active parameters
  ChemDriver& cd;

  const std::vector<Real>& TrueParameters() const {BL_ASSERT(true_parameters.size()>0); return true_parameters;}
  const std::vector<Real>& PriorMean() const {return prior_mean;}
  const std::vector<Real>& PriorSTD() const {return prior_std;}
  const std::vector<Real>& UpperBound() const {return upper_bound;}
  const std::vector<Real>& LowerBound() const {return lower_bound;}
  void setParamLowerBound( Real val, int idx );
  void setParamUpperBound( Real val, int idx );

  std::vector<Real> true_parameters;
  std::vector<Real> prior_mean;
  std::vector<Real> prior_std;
  std::vector<Real> lower_bound;
  std::vector<Real> upper_bound;
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
