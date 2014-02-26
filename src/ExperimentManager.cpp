#include <iostream>
#include <ExperimentManager.H>
#include <Rand.H>
#include <ParmParse.H>
  
ExperimentManager::ExperimentManager(ParameterManager& pmgr, ChemDriver& cd)
  : parameter_manager(pmgr), expts(PArrayManage), perturbed_data(0)
{
  ParmParse pp;
  int nExpts = pp.countval("experiments");
  Array<std::string> experiments;
  pp.getarr("experiments",experiments,0,nExpts);
  for (int i=0; i<nExpts; ++i) {
    std::string prefix = experiments[i];
    ParmParse ppe(prefix.c_str());
    std::string type; ppe.get("type",type);
    if (type == "CVReactor") {
      CVReactor *cv_reactor = new CVReactor(cd,experiments[i]);
      AddExperiment(cv_reactor,experiments[i]);
    }
    else if (type == "PREMIXReactor") {
      PREMIXReactor *premix_reactor = new PREMIXReactor(cd,experiments[i]);
      AddExperiment(premix_reactor,experiments[i]);
    }
    else {
      BoxLib::Abort("Unknown experiment type");
    }
  }
}

void
ExperimentManager::Clear()
{
  expts.clear(); expts.resize(0, PArrayManage);
  raw_data.clear();
  data_offsets.clear();
  num_expt_data = 0;
  expt_map.clear();
}

void
ExperimentManager::AddExperiment(SimulatedExperiment* expt,
                                 const std::string&   expt_id)
{
  int num_expts_old = expts.size();
  expts.resize(num_expts_old+1);
  expts.set(num_expts_old, expt);
  int num_new_values = expts[num_expts_old].NumMeasuredValues();
  raw_data.resize(num_expts_old+1,std::vector<Real>(num_new_values));
  data_offsets.resize(num_expts_old+1);
  data_offsets[num_expts_old] = ( num_expts_old == 0 ? 0 : 
                                  data_offsets[num_expts_old-1]+raw_data[num_expts_old-1].size() );
  num_expt_data = data_offsets[num_expts_old] + num_new_values;
  expt_map[expt_id] = num_expts_old;
}

void
ExperimentManager::InitializeExperiments()
{
  for (int i=0; i<expts.size(); ++i) {
    expts[i].InitializeExperiment();
  }
}

void
ExperimentManager::InitializeTrueData(const std::vector<Real>& true_parameters)
{
  GenerateTestMeasurements(true_parameters,true_data);

  std::cout << "ITD: " << true_data[0] << std::endl;

  true_std.resize(NumExptData());
  true_std_inv2.resize(NumExptData());
  for (int i=0; i<expts.size(); ++i) {
    expts[i].GetMeasurements(raw_data[i]);
    int offset = data_offsets[i];
    int nd = raw_data[i].size();
    for (int j=0; j<nd; ++j) {
      true_data[offset + j] = raw_data[i][j];
    }
    expts[i].GetMeasurementError(raw_data[i]);
    true_std_inv2.resize(nd);
    for (int j=0; j<nd; ++j) {
      true_std[offset + j] = raw_data[i][j];
      true_std_inv2[offset + j] = 1 / (true_std[offset + j] * true_std[offset + j]);
    }
  }    
  perturbed_data.resize(0);
}

void
ExperimentManager::GenerateExptData()
{
  BL_ASSERT(perturbed_data.size()==0);
  perturbed_data.resize(num_expt_data);
  BL_ASSERT(true_std.size() == num_expt_data);
  BL_ASSERT(true_data.size() == num_expt_data);
  // FIXME: Make more general
  for(int ii=0; ii<num_expt_data; ii++){
    Real small = true_std[ii];
    perturbed_data[ii] = std::max(small,true_data[ii] + true_std[ii] * randn());
  }
}

void
ExperimentManager::GenerateTestMeasurements(const std::vector<Real>& test_params,
                                            std::vector<Real>&       test_measurements)
{
  for (int i=0; i<test_params.size(); ++i) {
    parameter_manager[i] = test_params[i];      
  }
  test_measurements.resize(NumExptData());
  for (int i=0; i<expts.size(); ++i) {
    expts[i].GetMeasurements(raw_data[i]);
    int offset = data_offsets[i];

    int s = raw_data[i].size();
    for (int j=0; j<s; ++j) {
      test_measurements[offset + j] = raw_data[i][j];
    }
  }    
}

Real
ExperimentManager::ComputeLikelihood(const std::vector<Real>& test_data) const
{
  BL_ASSERT(test_data.size() == num_expt_data);
  if (perturbed_data.size()==0) {
    BoxLib::Abort("Must generate (perturbed) expt data before computing likelihood");
  }
  Real L = 0;
  for (int ii=0; ii<num_expt_data; ii++) {
    Real n = perturbed_data[ii] - test_data[ii];
    L += 0.5 * n * n * true_std_inv2[ii];
  }
  return L;
}

