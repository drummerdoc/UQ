#ifndef _ExperimentManager_
#define _ExperimentManager_

#include <ParameterManager.H>
#include <SimulatedExperiment.H>

struct ExperimentManager
{
  ExperimentManager(ParameterManager& pmgr, ChemDriver& cd, bool use_synthetic_data);
  
  void AddExperiment(SimulatedExperiment* expt,
                     const std::string& expt_id);

  void Clear();

  void SetVerbose(bool _verbose) {verbose = _verbose;}

  void InitializeExperiments();

  int size() const {return expts.size();}
  int NumExptData() const;

  void InitializeTrueData(const std::vector<Real>& true_parameters);

  void GenerateExptData();

  bool GenerateTestMeasurements(const std::vector<Real>& test_params,
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

  const SimulatedExperiment& Experiment(int i) const {return expts[i];}
  SimulatedExperiment& Experiment(int i) {return expts[i];} // Careful with this one!
  const std::vector<std::string>& ExperimentNames() {return expt_name;}

protected:
  void LogFailedCases(const std::vector<Real>& test_params,
                      const std::vector<Real>& test_measurements,
                      const std::vector<int>&  msgID);

  bool initialized, use_synthetic_data, verbose;
  ParameterManager& parameter_manager;
  PArray<SimulatedExperiment> expts;
  std::vector<std::vector<Real> > raw_data;
  std::vector<int> data_offsets;
  std::map<std::string,int> expt_map;
  std::vector<std::string> expt_name;
  
  int num_expt_data;
  std::vector<Real> true_data, perturbed_data;
  std::vector<Real> true_std;

  bool log_failed_cases;
  std::string log_folder_name;

private:
  ExperimentManager(const ExperimentManager& rhs);

};

#endif // _ExperimentManager_
