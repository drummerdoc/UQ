#include <iostream>
#include <ExperimentManager.H>
#include <Rand.H>
  
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
ExperimentManager::InitializeTrueData(const std::vector<Real>& _true_data,
                                      const std::vector<Real>& _true_data_std)
{
  BL_ASSERT(_true_data.size() == _true_data_std.size());
  num_expt_data = _true_data.size();
  true_data = _true_data;
  true_std = _true_data_std;
  true_std_inv2.resize(true_std.size());
  for (int i=0, N=true_std.size(); i<N; ++i) {
    BL_ASSERT(true_std[i] != 0);
    true_std_inv2[i] = 1 / (true_std[i] * true_std[i]);
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

