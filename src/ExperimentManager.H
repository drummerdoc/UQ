#ifndef _ExperimentManager_
#define _ExperimentManager_

#include <ParameterManager.H>
#include <SimulatedExperiment.H>

struct ExperimentManager
{
  ExperimentManager(ParameterManager& pmgr, ChemDriver& cd);
  
  void AddExperiment(SimulatedExperiment* expt,
                     const std::string& expt_id);

  void Clear();

  void InitializeExperiments();

  int NumExptData() const;

  void InitializeTrueData(const std::vector<Real>& true_parameters);

  void GenerateExptData();

  void GenerateTestMeasurements(const std::vector<Real>& test_params,
                                std::vector<Real>&       test_measurements);

  Real ComputeLikelihood(const std::vector<Real>& test_data) const;

  bool isgoodParamVal( Real, std::vector<Real>&, int );
  void get_param_limits( Real * kmin, Real * kmax, Real * ktyp, Real tol, 
                       std::vector<Real> & pvals, int idx);
  void get_param_interesting( Real * kmin, Real * kmax, Real * ktyp, Real tol, 
                       std::vector<Real> & pvals, int idx);

  const std::vector<Real>& TrueData() const;
  const std::vector<Real>& TrueDataWithObservationNoise() const;
  const std::vector<Real>& ObservationSTD() const;

  const SimulatedExperiment& Experiment(int i) const;

protected:  
  bool initialized;
  ParameterManager& parameter_manager;
  PArray<SimulatedExperiment> expts;
  std::vector<std::vector<Real> > raw_data;
  std::vector<int> data_offsets;
  std::map<std::string,int> expt_map;
  
  int num_expt_data;
  std::vector<Real> true_data, perturbed_data;
  std::vector<Real> true_std, true_std_inv2;

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
const std::vector<Real>&
ExperimentManager::TrueData() const
{
  return true_data;
}

inline
const std::vector<Real>&
ExperimentManager::ObservationSTD() const
{
  return true_std;
}

inline
const std::vector<Real>&
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
