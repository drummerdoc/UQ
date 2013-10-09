#include <iostream>

#include <ChemDriver.H>
#include <cminpack.h>
#include <SimulatedExperiment.H>

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

  Real ComputePrior(const Array<Real>& params) const {
    BL_ASSERT(params.size() == NumParams());
    Real p = 0;
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

  Real TypicalValue(int i) {return 1;} // FIXME: Generalize this

  bool prior_stats_initialized;
  PArray<ChemDriver::Parameter> active_parameters; // The set of active parameters
  ChemDriver& cd;

  Array<Real> true_prior;
  Array<Real> prior_mean;
  Array<Real> prior_std;
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
    data_offsets[num_expts_old] = ( num_expts_old == 0 ? 0 : 
                                    data_offsets[num_expts_old-1]+raw_data[num_expts_old-1].size() );
    num_expt_data = data_offsets[num_expts_old] + num_new_values;
    expt_map[expt_id] = num_expts_old;
  }

  void InitializeExperiments() {
    for (int i=0; i<expts.size(); ++i) {
      expts[i].InitializeExperiment();
    }
  }

  int NumExptData() const {return num_expt_data;}
  void InitializeTrueData(const Array<Real>& _true_data,
                          const Array<Real>& _true_data_std) {
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

  Real ComputeLikelihood(const Array<Real>& test_data) const {
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

  const Array<Real>& TrueData() const {return true_data;}
  const Array<Real>& TrueDataWithObservationNoise() const {return perturbed_data;}

  
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


struct MINPACKstruct
{
  MINPACKstruct(ChemDriver& cd, Real _param_eps)
  : parameter_manager(cd), expt_manager(parameter_manager), param_eps(_param_eps) {}

  void ResizeWork() {
    int N = parameter_manager.NumParams();
    int NWORK = 2;
    work.resize(NWORK);
    for (int i=0; i<NWORK; ++i) {
      work[i].resize(N);
    }
  }

  ParameterManager parameter_manager;
  ExperimentManager expt_manager;
  Array<Array<Real> > work;
  Real param_eps;
};

static ChemDriver cd;
Real param_eps = 1.5e-8;
MINPACKstruct mystruct(cd,param_eps);

Real 
funcF(void* p, const Array<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  Array<Real> dvals(s->expt_manager.NumExptData());
  s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  return s->parameter_manager.ComputePrior(pvals) + s->expt_manager.ComputeLikelihood(dvals);
}

// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
Real
der_cfd(void* p, const Array<Real>& X, int K) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  Array<Real>& xdX1 = s->work[0];
  Array<Real>& xdX2 = s->work[1];

  int num_vals = s->parameter_manager.NumParams();

  for (int ii=0; ii<num_vals; ii++){
    xdX1[ii] = X[ii];
    xdX2[ii] = X[ii];
  }
                
  Real typ = std::max(s->parameter_manager.TypicalValue(K), std::abs(X[K]));
  Real h = typ * s->param_eps;

  xdX1[K] += h;
  Real fx1 = funcF(p, xdX1);

  xdX2[K] -= h;
  Real fx2 = funcF(p, xdX2);

  return (fx1-fx2)/(xdX1[K]-xdX2[K]);
}

// Compute the derivative of the function funcF with respect to 
// the Kth variable (forward finite differences)
Real
der_ffd(void* p, const Array<Real>& X, int K) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  Array<Real>& xdX = s->work[0];

  int num_vals = s->parameter_manager.NumParams();

  for (int ii=0; ii<num_vals; ii++){
    xdX[ii]  = X[ii];
  }
                
  Real typ = std::max(s->parameter_manager.TypicalValue(K), std::abs(xdX[K]));
  Real h = typ * s->param_eps;

  xdX[K] += h;
  Real fx1 = funcF(p, xdX);
  Real fx2 = funcF(p, X);

  Real der = (fx1-fx2)/(xdX[K]-X[K]);
  return der;
}

// Gradient with finite differences
void grad(void * p, const Array<Real>& X, Array<Real>& gradF) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  for (int ii=0;ii<num_vals;ii++){
    gradF[ii] = der_cfd(p,X,ii); 
  } 
}

// This is what we give to MINPACK
int FCN(void       *p,    
        int	   NP,
        const Real *X,
        Real       *FVEC,
        int 	   IFLAGP)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  s->ResizeWork();
  Array<Real> Xv(X,NP);
  Array<Real> Fv(FVEC,NP);
  grad(p,Xv,Fv);
  return 0;
}

// Call minpack
void minimize(void *p, const Array<Real>& guess, Array<Real>& soln)
{
  int INFO,LWA=180;
  Real TOL=1e-14;
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  Array<Real> FVEC(num_vals);
  Array<Real> WA(LWA);
  soln = guess;
  INFO = hybrd1(FCN,p,num_vals,soln.dataPtr(),FVEC.dataPtr(),TOL,WA.dataPtr(),WA.size());   
  std::cout << "minpack INFO: " << INFO << std::endl;
};



int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  if (argc<2) print_usage(argc,argv);

  ParameterManager& parameter_manager = mystruct.parameter_manager;
  ExperimentManager& expt_manager = mystruct.expt_manager;  
  

  Array<Real> true_params;
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_BETA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_EA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_A));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_BETA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_EA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_A));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TS));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TSSS));

  true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_A));
  true_params.push_back(parameter_manager.AddParameter(9,ChemDriver::FWD_A));
  true_params.push_back(parameter_manager.AddParameter(10,ChemDriver::FWD_A));
  //true_params.push_back(parameter_manager.AddParameter(11,ChemDriver::FWD_A));
  //true_params.push_back(parameter_manager.AddParameter(12,ChemDriver::FWD_A));
  //true_params.push_back(parameter_manager.AddParameter(13,ChemDriver::FWD_A));
  //true_params.push_back(parameter_manager.AddParameter(14,ChemDriver::FWD_A));
  int num_params = parameter_manager.NumParams();
  std::cout << "NumParams:" << num_params << std::endl; 
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  True: " << parameter_manager[ii] << std::endl;
  }

  CVReactor cv_reactor(cd);
  CVReactor cv_reactor2(cd);
  expt_manager.AddExperiment(cv_reactor,"exp1");
  //expt_manager.AddExperiment(cv_reactor2,"exp2");
  expt_manager.InitializeExperiments();

  int num_data = expt_manager.NumExptData();
  std::cout << "NumData:" << num_data << std::endl; 
  Array<Real> true_data(num_data);
  Array<Real> true_data_std(num_data,50); // Set variance of data 

  expt_manager.GenerateTestMeasurements(true_params,true_data);
  expt_manager.InitializeTrueData(true_data,true_data_std);

  expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)
  const Array<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();

  std::cout << "True and noisy data:\n"; 
  for(int ii=0; ii<num_data; ii++){
    std::cout << "  True: " << true_data[ii]
              << "  Noisy: " << perturbed_data[ii]
              << "  Standard deviation: " << true_data_std[ii] << std::endl;
  }


  Array<Real> prior_mean(num_params);
  Array<Real> prior_std(num_params);
  for(int ii=0; ii<num_params; ii++){
    prior_std[ii] = 0.5;
    prior_std[ii] = 0.8;
    if (prior_std[ii] == 0) {prior_std[ii] = 1e-2;}

    prior_mean[ii] = true_params[ii]*(1+0.5*prior_std[ii]);
    if (prior_mean[ii] == 0) {prior_mean[ii] =1e-2;}
  }

  std::cout << "True and prior mean:\n"; 
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  True: " << true_params[ii]
              << "  Prior: " << prior_mean[ii]
              << "  Standard deviation: " << prior_std[ii] << std::endl;
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

  // Call minpack
  Array<Real> soln(num_params);
  minimize((void*)(&mystruct), prior_mean, soln);
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  Final: " << parameter_manager[ii] << std::endl;
  }

  BoxLib::Finalize();
}

