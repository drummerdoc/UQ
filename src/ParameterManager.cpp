#include <iostream>
#include <map>

#include <ParameterManager.H>
#include <Rand.H>
#include <ParmParse.H>

ParameterManager::ParameterManager(ChemDriver& _cd)
  : cd(_cd)
{
  prior_stats_initialized = false;

  std::map<std::string,ChemDriver::REACTION_PARAMETER> PTypeMap;
  PTypeMap["FWD_A"]     = ChemDriver::FWD_A;
  PTypeMap["FWD_BETAA"] = ChemDriver::FWD_BETA;
  PTypeMap["FWD_EA"]    = ChemDriver::FWD_EA;
  PTypeMap["LOW_A"]     = ChemDriver::LOW_A;
  PTypeMap["LOW_BETAA"] = ChemDriver::LOW_BETA;
  PTypeMap["LOW_EA"]    = ChemDriver::LOW_EA;
  PTypeMap["REV_A"]     = ChemDriver::REV_A;
  PTypeMap["REV_BETAA"] = ChemDriver::REV_BETA;
  PTypeMap["REV_EA"]    = ChemDriver::REV_EA;
  PTypeMap["TROE_A"]    = ChemDriver::TROE_A;
  PTypeMap["TROE_A"]    = ChemDriver::TROE_A;
  PTypeMap["TROE_TS"]   = ChemDriver::TROE_TS;
  PTypeMap["TROE_TSS"]  = ChemDriver::TROE_TSS;
  PTypeMap["TROE_TSSS"] = ChemDriver::TROE_TSSS;
  PTypeMap["SRI_A"]     = ChemDriver::SRI_A;
  PTypeMap["SRI_B"]     = ChemDriver::SRI_B;
  PTypeMap["SRI_C"]     = ChemDriver::SRI_C;
  PTypeMap["SRI_D"]     = ChemDriver::SRI_D;
  PTypeMap["SRI_E"]     = ChemDriver::SRI_E;

  ParmParse pp;
  Array<std::string> parameters;
  int np = pp.countval("parameters");
  BL_ASSERT(np>0);
  pp.getarr("parameters",parameters,0,np);

  std::vector<Real> lower_bound(np);
  std::vector<Real> upper_bound(np);
  std::vector<Real> prior_mean(np);
  std::vector<Real> prior_std(np);

  for (int i=0; i<np; ++i) {
    std::string prefix = parameters[i];
    ParmParse ppp(prefix.c_str());
    int reaction_id; ppp.get("reaction_id",reaction_id);
    if (reaction_id<0 || reaction_id > cd.numReactions()) {
      BoxLib::Abort("Reaction ID invalid");
    }

    std::string type; ppp.get("type",type);
    std::map<std::string,ChemDriver::REACTION_PARAMETER>::const_iterator it = PTypeMap.find(type);
    if (it == PTypeMap.end()) {
      BoxLib::Abort("Unrecognized reaction parameter");
    }
    AddParameter(reaction_id,it->second);

    ppp.get("prior_mean",prior_mean[i]);
    ppp.get("prior_std",prior_std[i]);
    ppp.get("lower_bound",lower_bound[i]);
    ppp.get("upper_bound",upper_bound[i]);
  }
  SetStatsForPrior(prior_mean,prior_std,lower_bound,upper_bound);
}

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


void ParameterManager::setParamLowerBound( Real val, int idx ){
    lower_bound[idx] = val;
}
void ParameterManager::setParamUpperBound( Real val, int idx ){
    upper_bound[idx] = val;
}
