#include <iostream>

#include <ChemDriver.H>

static
void 
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " pmf_file=<input fab file name> [options] \n";
  std::cerr << "\tOptions:       Patm = <pressure, in atmospheres> \n";
  std::cerr << "\t                 dt = <time interval, in seconds> \n";
  std::cerr << "\t              Tfile = <T search value, in K> \n";
  exit(1);
}

// ******************************************************
// Generate uniformly distributed random number 
static
Real drand() {
  return (rand()+1.0)/(RAND_MAX+1.0);
}
  
// Generate standard normal random number 
static
Real randn(){
  double pi; 
  pi =  3.14159265358979323846;
  return sqrt(-2*log(drand())) * cos(2*pi*drand());
}


typedef PArray<ChemDriver::Parameter> Parameters;

struct SimulatedExperiment
{
  SimulatedExperiment() {}
  SimulatedExperiment(const SimulatedExperiment& rhs);
  ~SimulatedExperiment() {}
  virtual void GetMeasurements(Array<Real>& simulated_observations) = 0;
  virtual int NumMeasuredValues() const = 0;
  virtual void InitializeExperiment() = 0;
};


struct CVReactor
  : public SimulatedExperiment
{
  CVReactor() {
    Real dt = 1;
    int num_time_intervals = 10;
    measurement_times.resize(num_time_intervals+1);
    for (int i=0; i<=num_time_intervals; ++i) {
      measurement_times[i] = i*dt/num_time_intervals;
    }
    num_measured_values = measurement_times.size();
  }
  CVReactor(const CVReactor& rhs) {
    measurement_times = rhs.measurement_times;
    measured_comps = rhs.measured_comps;
    num_measured_values = num_measured_values;
    s_init.resize(rhs.s_init.box(),rhs.s_init.nComp()); s_init.copy(rhs.s_init);
    s_final.resize(rhs.s_final.box(),rhs.s_final.nComp()); s_final.copy(rhs.s_final);
    C_0.resize(rhs.C_0.box(),rhs.C_0.nComp()); C_0.copy(rhs.C_0);
    I_R.resize(rhs.I_R.box(),rhs.I_R.nComp()); I_R.copy(rhs.I_R);
    funcCnt.resize(rhs.funcCnt.box(),rhs.funcCnt.nComp()); funcCnt.copy(rhs.funcCnt);
    sCompY=rhs.sCompY;
    sCompT=rhs.sCompT;
    sCompR=rhs.sCompR;
    sCompRH=rhs.sCompRH;
    Patm=rhs.Patm;
  }
  ~CVReactor() {}
  virtual void GetMeasurements(Array<Real>& simulated_observations) {
    for (int i=0; i<num_measured_values; ++i) {
      simulated_observations[i] = i;
    }
  }
  int NumMeasuredValues() const {return num_measured_values;}
  virtual void InitializeExperiment() {
  }

private:
  // Compute observation from evolution
  Real FinalValue() const {return 0;}

  Array<Real> measurement_times;
  Array<int> measured_comps;
  int num_measured_values;

  FArrayBox s_init, s_final, C_0, I_R, funcCnt; 
  int sCompY, sCompT, sCompR, sCompRH;
  Real Patm;
};

struct ParameterManager
{
  ParameterManager(ChemDriver& _cd) : cd(_cd) {prior_stats_initialized = false;}

  // Add parameter to active set, return default value
  Real AddParameter(int reaction, const ChemDriver::REACTION_PARAMETER& rp) {
    int len = active_parameters.size();
    active_parameters.resize(len+1,PArrayManage);
    active_parameters.set(len, new ChemDriver::Parameter(reaction,rp));
    return active_parameters[len].DefaultValue();
  }

  // Reset internal data back to state of initialization
  void ResetParametersToDefault();
  int NumParams() const {return active_parameters.size();}
  void SetStatsForPrior(const Array<Real>& mean,
                        const Array<Real>& std) {
    prior_mean = mean;
    prior_std = std;
    prior_stats_initialized = true;
  }

  void GenerateSampleOfPrior(Array<Real>& parameter_samples) const {
    BL_ASSERT(prior_stats_initialized);
    int num_vals = NumParams();
    parameter_samples.resize(num_vals);
    for(int ii=0; ii<num_vals; ii++){
      parameter_samples[ii] = prior_mean[ii] + prior_std[ii] * randn();
    }
  }

  Real ComputePrior(const Array<double>& params) const {
    BL_ASSERT(params.size() == NumParams());
    double p = 0;
    for (int ii=0, End=NumParams(); ii<End; ii++){
      p+=(prior_mean[ii]-params[ii])*(prior_mean[ii]-params[ii])/2/prior_std[ii]/prior_std[ii];
    }
    return p;
  }

  ChemDriver::Parameter& operator[](int i) {
    return active_parameters[i];
  }

  const ChemDriver::Parameter& operator[](int i) const {
    return active_parameters[i];
  }

  bool prior_stats_initialized;
  PArray<ChemDriver::Parameter> active_parameters; // The set of active parameters
  ChemDriver& cd;

  Array<double> true_prior;
  Array<double> prior_mean;
  Array<double> prior_std;
};




struct ExperimentManager
{
  ExperimentManager(ParameterManager& pmgr) : parameter_manager(pmgr), perturbed_data(0) {}
  
  void AddExperiment(SimulatedExperiment& expt,
                     const std::string& expt_id) {
    int num_expts_old = expts.size();
    expts.resize(num_expts_old+1,PArrayNoManage);
    expts.set(num_expts_old, &expt);
    int num_new_values = expts[num_expts_old].NumMeasuredValues();
    raw_data.resize(num_expts_old+1,Array<Real>(num_new_values));
    data_offsets.resize(num_expts_old+1);
    data_offsets[num_expts_old] = ( num_expts_old == 0 ? 0 : data_offsets[num_expts_old-1]+raw_data[num_expts_old-1].size() );
    num_expt_data = data_offsets[num_expts_old] + num_new_values;
    expt_map[expt_id] = num_expts_old;
  }

  void InitializeExperiments() {
    for (int i=0; i<expts.size(); ++i) {
      expts[i].InitializeExperiment();
    }
  }

  int NumExptData() const {return num_expt_data;}
  void InitializeTrueData(const Array<double>& _true_data,
                          const Array<double>& _true_data_std) {
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

  void GenerateExptData() {
    perturbed_data.resize(num_expt_data);
    BL_ASSERT(true_std.size() == num_expt_data);
    BL_ASSERT(true_data.size() == num_expt_data);
    for(int ii=0; ii<num_expt_data; ii++){
      perturbed_data[ii] = true_data[ii] + true_std[ii] * randn();
    }
  }

  void GenerateTestMeasurements(const Array<Real>& test_params,
                                Array<Real>&       test_measurements) {
    for (int i=0; i<test_params.size(); ++i) {
      parameter_manager[i] = test_params[i];      
    }
    for (int i=0; i<expts.size(); ++i) {
      expts[i].GetMeasurements(raw_data[i]);
      int offset = data_offsets[i];
      for (int j=0; j<raw_data[i].size(); ++j) {
        test_measurements[offset + j] = raw_data[i][j];
      }
    }    
  }

  Real ComputeLikelihood(const Array<double>& test_data) const {
    BL_ASSERT(test_data.size() == num_expt_data);
    if (perturbed_data.size()==0) {
      BoxLib::Abort("Must generate (perturbed) expt data before computing likelihood");
    }
    double L = 0;
    for (int ii=0; ii<num_expt_data; ii++) {
      double n = perturbed_data[ii] - test_data[ii];
      L += 0.5 * n * n * true_std_inv2[ii];
    }
    return L;
  }

  const Array<Real>& TrueData() const {return true_data;}
  const Array<Real>& TrueDataWithObservationNoise() const {return perturbed_data;}

  
  bool initialized;
  ParameterManager& parameter_manager;
  PArray<SimulatedExperiment> expts;
  Array<Array<Real> > raw_data;
  Array<int> data_offsets;
  std::map<std::string,int> expt_map;
  
  int num_expt_data;
  Array<double> true_data, perturbed_data;
  Array<double> true_std, true_std_inv2;
};


struct MINPACKstruct
{
  MINPACKstruct(ChemDriver& cd) : parameter_manager(cd), expt_manager(parameter_manager) {}
  ParameterManager parameter_manager;
  ExperimentManager expt_manager;  
};

static ChemDriver cd;
MINPACKstruct mystruct(cd);

Real 
funcF(void* p, const Array<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  Array<Real> dvals(s->expt_manager.NumExptData());
  s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  return s->parameter_manager.ComputePrior(pvals) + s->expt_manager.ComputeLikelihood(dvals);
}

int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  if (argc<2) print_usage(argc,argv);

  ParameterManager& parameter_manager = mystruct.parameter_manager;
  ExperimentManager expt_manager = mystruct.expt_manager;  
  

  Array<Real> true_params;
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_BETA));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_EA));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_A));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_BETA));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_EA));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_A));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TS));
  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TSSS));
  int num_params = parameter_manager.NumParams();
  std::cout << "NumParams:" << num_params << std::endl; 
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  True: " << parameter_manager[ii] << std::endl;
  }

  CVReactor cv_reactor;
  CVReactor cv_reactor2;
  expt_manager.AddExperiment(cv_reactor,"exp1");
  expt_manager.AddExperiment(cv_reactor2,"exp2");
  expt_manager.InitializeExperiments();

  int num_data = expt_manager.NumExptData();
  std::cout << "NumData:" << num_data << std::endl; 
  Array<Real> true_data(num_data);
  Array<Real> true_data_std(num_data,2); // Set variance of data (likelihood) to 14

  expt_manager.GenerateTestMeasurements(true_params,true_data);
  expt_manager.InitializeTrueData(true_data,true_data_std);

  // Output true data and std
  std::cout << "True data:\n"; 
  for(int ii=0; ii<num_data; ii++){
    std::cout << "  True: " << true_data[ii] << std::endl;
  }

  expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)
  const Array<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();

  std::cout << "True data with observation noise:\n"; 
  for(int ii=0; ii<num_data; ii++){
    std::cout << "  Noisy: " << perturbed_data[ii] << "  Standard deviation: " << true_data_std[ii] << std::endl;
  }


  Array<Real> prior_mean(num_params);
  Array<Real> prior_std(num_params);
  for(int ii=0; ii<num_params; ii++){
    prior_std[ii] = true_params[ii]*0.1;
    if (prior_std[ii] == 0) {prior_std[ii] =1e-2;}

    prior_mean[ii] = true_params[ii]*(1+0.5*prior_std[ii]);
    if (prior_mean[ii] == 0) {prior_mean[ii] =1e-2;}
  }

  Array<Real> prior_data(num_data);
  std::cout << "Data with prior mean:\n"; 
  expt_manager.GenerateTestMeasurements(prior_mean,prior_data);
  for(int ii=0; ii<num_data; ii++){
    std::cout << "  Data with prior: " << prior_data[ii] << std::endl;
  }


  parameter_manager.SetStatsForPrior(prior_mean,prior_std);

  Real F = funcF((void*)(&mystruct), prior_mean);  	
  std::cout << "F = " << F << std::endl;


  BoxLib::Finalize();
}

