#ifndef _ExperimentManager_
#define _ExperimentManager_

#include <ParameterManager.H>
#include <SimulatedExperiment.H>

struct ExperimentManager
{
  ExperimentManager(ParameterManager& pmgr) : parameter_manager(pmgr), perturbed_data(0) {}
  
  void AddExperiment(SimulatedExperiment& expt,
                     const std::string& expt_id);

  void InitializeExperiments();

  int NumExptData() const;

  void InitializeTrueData(const Array<Real>& _true_data,
                          const Array<Real>& _true_data_std);

  void GenerateExptData();

  void GenerateTestMeasurements(const Array<Real>& test_params,
                                Array<Real>&       test_measurements);

  Real ComputeLikelihood(const Array<Real>& test_data) const;

  const Array<Real>& TrueData() const;
  const Array<Real>& TrueDataWithObservationNoise() const;

  const SimulatedExperiment& Experiment(int i) const;

protected:  
  bool initialized;
  ParameterManager& parameter_manager;
  PArray<SimulatedExperiment> expts;
  Array<Array<Real> > raw_data;
  Array<int> data_offsets;
  std::map<std::string,int> expt_map;
  
  int num_expt_data;
  Array<Real> true_data, perturbed_data;
  Array<Real> true_std, true_std_inv2;

private:
  ExperimentManager(const ExperimentManager& rhs);

};

inline
int
ExperimentManager::NumExptData() const
{
  return num_expt_data;
}

inline
const Array<Real>&
ExperimentManager::TrueData() const
{
  return true_data;
}

inline
const Array<Real>&
ExperimentManager::TrueDataWithObservationNoise() const
{
  return perturbed_data;
}

inline
const
SimulatedExperiment&
ExperimentManager::Experiment(int i) const
{
  return expts[i];
}


#endif // _ExperimentManager_
