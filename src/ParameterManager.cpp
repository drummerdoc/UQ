#include <iostream>
#include <map>

#include <ParameterManager.H>
#include <Rand.H>
#include <ParmParse.H>

ParameterManager::ParameterManager(ChemDriver& _cd)
  : cd(_cd)
{
  prior_stats_initialized = false;

  std::map<std::string,REACTION_PARAMETER> PTypeMap;
  PTypeMap["FWD_A"]      = FWD_A;
  PTypeMap["FWD_BETAA"]  = FWD_BETA;
  PTypeMap["FWD_EA"]     = FWD_EA;
  PTypeMap["LOW_A"]      = LOW_A;
  PTypeMap["LOW_BETAA"]  = LOW_BETA;
  PTypeMap["LOW_EA"]     = LOW_EA;
  PTypeMap["REV_A"]      = REV_A;
  PTypeMap["REV_BETAA"]  = REV_BETA;
  PTypeMap["REV_EA"]     = REV_EA;
  PTypeMap["TROE_A"]     = TROE_A;
  PTypeMap["TROE_A"]     = TROE_A;
  PTypeMap["TROE_TS"]    = TROE_TS;
  PTypeMap["TROE_TSS"]   = TROE_TSS;
  PTypeMap["TROE_TSSS"]  = TROE_TSSS;
  PTypeMap["SRI_A"]      = SRI_A;
  PTypeMap["SRI_B"]      = SRI_B;
  PTypeMap["SRI_C"]      = SRI_C;
  PTypeMap["SRI_D"]      = SRI_D;
  PTypeMap["SRI_E"]      = SRI_E;
  PTypeMap["THIRD_BODY"] = THIRD_BODY;

  ParmParse pp;
  Array<std::string> parameters;
  int np = pp.countval("parameters");
  BL_ASSERT(np>0);
  pp.getarr("parameters",parameters,0,np);

  std::vector<Real> lower_bound(np);
  std::vector<Real> upper_bound(np);
  std::vector<Real> prior_mean(np);
  std::vector<Real> prior_std(np);
  std::vector<Real> ensemble_std(np);

  bool requires_sync = false; // If there are dependent parameters, we will need to sync them before continuing

  for (int i=0; i<np; ++i) {
    std::string prefix = parameters[i];
    ParmParse ppp(prefix.c_str());
    int reaction_id; ppp.get("reaction_id",reaction_id);
    if (reaction_id<0 || reaction_id > cd.numReactions()) {
      BoxLib::Abort("Reaction ID invalid");
    }

    std::string type; ppp.get("type",type);
    std::map<std::string,REACTION_PARAMETER>::const_iterator it = PTypeMap.find(type);
    if (it == PTypeMap.end()) {
      BoxLib::Abort("Unrecognized reaction parameter");
    }

    int id = -1;
    if (type == "THIRD_BODY") {
      std::string tb_name; ppp.get("tb_name",tb_name);
      id = cd.index(tb_name);
      BL_ASSERT(id >= 0);
    }
    AddParameter(reaction_id,it->second,id);

    int ndp = ppp.countval("dependent_parameters");
    if (ndp > 0) {
      dependent_parameters[i].resize(ndp);
      Array<std::string> dpnames; ppp.getarr("dependent_parameters",dpnames,0,ndp);
      for (int j=0; j<ndp; ++j) {

	std::string dplist = prefix + ".dependent_parameters." + dpnames[j];
	ParmParse pppd(dplist.c_str());

	int dpreaction_id; pppd.get("reaction_id",dpreaction_id);
	if (dpreaction_id<0 || dpreaction_id > cd.numReactions()) {
	  BoxLib::Abort("Dependent reaction ID invalid");
	}

	std::string dptype; pppd.get("type",dptype);
	std::map<std::string,REACTION_PARAMETER>::const_iterator it = PTypeMap.find(dptype);
	if (it == PTypeMap.end()) {
	  BoxLib::Abort("Unrecognized dependent reaction parameter");
	}

	int did = -1;
	if (dptype == "THIRD_BODY") {
	  std::string dtb_name; pppd.get("tb_name",dtb_name);
	  did = cd.index(dtb_name);
	  BL_ASSERT(id >= 0);
	}

	dependent_parameters[i].set(j, new ChemDriver::Parameter(dpreaction_id,it->second,did));
	requires_sync = true;
      }
    }

    ppp.get("prior_mean",prior_mean[i]);
    ppp.get("prior_std",prior_std[i]);
    ppp.get("ensemble_std",ensemble_std[i]);
    ppp.get("lower_bound",lower_bound[i]);
    ppp.get("upper_bound",upper_bound[i]);
  }
  SetStatsForPrior(prior_mean,prior_std, ensemble_std,lower_bound,upper_bound);

  if (requires_sync) {
    for (int i=0; i<active_parameters.size(); ++i) {
      SetParameter(i,GetParameterDefault(i));
    }

#if 0
    for (int i=0; i<active_parameters.size(); ++i) {
      for (int j=0; j<dependent_parameters[i].size(); ++j) {
	if (j==0) {
	  std::cout << "For parameter: " << active_parameters[i] << " " << std::endl;
	}
	std::cout << "    added dependent parameter: " << dependent_parameters[i][j] << std::endl;
      }
    }
#endif
  }
}

// Add parameter to active set, return default value
Real
ParameterManager::AddParameter(int                       reaction,
                               const REACTION_PARAMETER& rp,
                               int                       species_id)
{
  int len = active_parameters.size();
  active_parameters.resize(len+1,PArrayManage);
  active_parameters.set(len, new ChemDriver::Parameter(reaction,rp,species_id));
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
ParameterManager::SetParameter(int i, Real val)
{
  BL_ASSERT(i>=0 && i<active_parameters.size());
  active_parameters[i].Value() = val;
  for (int j=0; j<dependent_parameters[i].size(); ++j) {
    dependent_parameters[i][j].Value() = val;
  }
}

Real
ParameterManager::GetParameterCurrent(int i) const
{
  BL_ASSERT(i>=0 && i<active_parameters.size());
  return active_parameters[i].Value();
}

Real
ParameterManager::GetParameterDefault(int i) const
{
  BL_ASSERT(i>=0 && i<active_parameters.size());
  return active_parameters[i].DefaultValue();
}

Real
ParameterManager::GetParameterTypical(int i) const
{
  BL_ASSERT(i>=0 && i<active_parameters.size());
  return active_parameters[i].DefaultValue();
}

void
ParameterManager::ResetParametersToDefault()
{
  for (int i=0, End=active_parameters.size(); i<End; ++i) {
    SetParameter(i,GetParameterDefault(i));
  }
}

void
ParameterManager::Clear()
{
  ResetParametersToDefault();
  for (int i=0; i<active_parameters.size(); ++i) {
    dependent_parameters[i].clear();
  }
  active_parameters.clear();
  prior_stats_initialized = false;
}

void
ParameterManager::SetStatsForPrior(const std::vector<Real>& _mean,
                                   const std::vector<Real>& _prior_std,
                                   const std::vector<Real>& _ensemble_std,
                                   const std::vector<Real>& _lower_bound,
                                   const std::vector<Real>& _upper_bound)
{
  prior_mean = _mean;
  prior_std = _prior_std;
  ensemble_std = _ensemble_std;
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
