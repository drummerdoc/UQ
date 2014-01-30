#include <iostream>

#include <ParameterManager.H>
#include <Rand.H>

// Add parameter to active set, return default value
Real
ParameterManager::AddParameter(int reaction, const ChemDriver::REACTION_PARAMETER& rp)
{
  int len = active_parameters.size();
  active_parameters.resize(len+1,PArrayManage);
  active_parameters.set(len, new ChemDriver::Parameter(reaction,rp));
  prior_stats_initialized = false;
  true_parameters.resize(len+1);
  true_parameters[len] = active_parameters[len].DefaultValue();
  return true_parameters[len];
}

int
ParameterManager::NumParams() const
{
  return active_parameters.size();
}

void
ParameterManager::ResetParametersToDefault()
{
  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    active_parameters[i] = active_parameters[i].DefaultValue();
  }
}

void
ParameterManager::Clear()
{
  ResetParametersToDefault();
  active_parameters.clear();
  active_parameters.resize(0);
  prior_stats_initialized = false;
}

void
ParameterManager::SetStatsForPrior(const std::vector<Real>& _mean,
                                   const std::vector<Real>& _std,
                                   const std::vector<Real>& _lower_bound,
                                   const std::vector<Real>& _upper_bound)
{
  prior_mean = _mean;
  prior_std = _std;
  lower_bound = _lower_bound;
  upper_bound = _upper_bound;
  prior_stats_initialized = true;
}

void
ParameterManager::GenerateSampleOfPrior(std::vector<Real>& parameter_samples) const
{
  BL_ASSERT(prior_stats_initialized);
  int num_vals = NumParams();
  parameter_samples.resize(num_vals);
  for(int ii=0; ii<num_vals; ii++){
    parameter_samples[ii] = prior_mean[ii] + prior_std[ii] * randn();
  }
}

std::pair<bool,Real>
ParameterManager::ComputePrior(const std::vector<Real>& params) const
{
  BL_ASSERT(prior_stats_initialized);
  BL_ASSERT(params.size() == NumParams());
  Real p = 0;
  bool sample_oob = false;
  for (int ii=0, End=NumParams(); ii<End; ii++){
    sample_oob |= (params[ii] < lower_bound[ii] || params[ii] > upper_bound[ii]);
    p+=(prior_mean[ii]-params[ii])*(prior_mean[ii]-params[ii])/2/prior_std[ii]/prior_std[ii];
  }
  bool sample_ok = !sample_oob;
  return std::pair<bool,Real>(sample_ok,p);
}

