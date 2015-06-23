#ifndef _ParameterManager_
#define _ParameterManager_

#include <ChemDriver.H>

struct ParameterManager
{
  ParameterManager(ChemDriver& _cd);

  // Add parameter to active set, return default value
  Real AddParameter(int                       reaction,
                    const REACTION_PARAMETER& rp,
                    int                       species_id);
  void Clear();

  // Reset internal data back to state of initialization
  void ResetParametersToDefault();

  int NumParams() const;

  void SetStatsForPrior(const std::vector<Real>& mean,
                        const std::vector<Real>& prior_std,
                        const std::vector<Real>& ensemble_std,
                        const std::vector<Real>& lower_bound,
                        const std::vector<Real>& upper_bound);

  void GenerateSampleOfPrior(std::vector<Real>& parameter_samples) const;

  std::pair<bool,Real> ComputePrior(const std::vector<Real>& params) const;

  void SetParameter(int i, Real val);
  Real GetParameterCurrent(int i) const;
  Real GetParameterTypical(int i) const;
  Real GetParameterDefault(int i) const;

  bool prior_stats_initialized;
  PArray<ChemDriver::Parameter> active_parameters; // The set of active parameters
  ChemDriver& cd;

  const std::vector<Real>& TrueParameters() const {BL_ASSERT(true_parameters.size()>0); return true_parameters;}
  const std::vector<Real>& PriorMean() const {return prior_mean;}
  const std::vector<Real>& PriorSTD() const {return prior_std;}
  const std::vector<Real>& EnsembleSTD() const {return ensemble_std;}
  const std::vector<Real>& UpperBound() const {return upper_bound;}
  const std::vector<Real>& LowerBound() const {return lower_bound;}
  void setParamLowerBound( Real val, int idx );
  void setParamUpperBound( Real val, int idx );

  std::vector<Real> true_parameters;
  std::vector<Real> prior_mean;
  std::vector<Real> prior_std;
  std::vector<Real> ensemble_std;
  std::vector<Real> lower_bound;
  std::vector<Real> upper_bound;

  std::map<int,PArray<ChemDriver::Parameter> > dependent_parameters;
};

#endif // ParameterManager
