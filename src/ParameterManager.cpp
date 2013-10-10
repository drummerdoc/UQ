
#include <ParameterManager.H>
#include <Rand.H>

// Add parameter to active set, return default value
Real
ParameterManager::AddParameter(int reaction, const ChemDriver::REACTION_PARAMETER& rp)
{
  int len = active_parameters.size();
  active_parameters.resize(len+1,PArrayManage);
  active_parameters.set(len, new ChemDriver::Parameter(reaction,rp));
  return active_parameters[len].DefaultValue();
}

void
ParameterManager::SetStatsForPrior(const Array<Real>& mean,
                                   const Array<Real>& std) {
  prior_mean = mean;
  prior_std = std;
  prior_stats_initialized = true;
}

void
ParameterManager::GenerateSampleOfPrior(Array<Real>& parameter_samples) const
{
  BL_ASSERT(prior_stats_initialized);
  int num_vals = NumParams();
  parameter_samples.resize(num_vals);
  for(int ii=0; ii<num_vals; ii++){
    parameter_samples[ii] = prior_mean[ii] + prior_std[ii] * randn();
  }
}

Real
ParameterManager::ComputePrior(const Array<Real>& params) const
{
  BL_ASSERT(prior_stats_initialized);
  BL_ASSERT(params.size() == NumParams());
  Real p = 0;
  for (int ii=0, End=NumParams(); ii<End; ii++){
    p+=(prior_mean[ii]-params[ii])*(prior_mean[ii]-params[ii])/2/prior_std[ii]/prior_std[ii];
  }
  return p;
}

