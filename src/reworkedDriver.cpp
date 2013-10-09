#include <iostream>

#include <ChemDriver.H>
#include <cminpack.h>

#include <ExperimentManager.H>

static
void 
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " pmf_file=<input fab file name> [options] \n";
  std::cerr << "\tOptions:       Patm = <pressure, in atmospheres> \n";
  std::cerr << "\t                 dt = <time interval, in seconds> \n";
  std::cerr << "\t     time_intervals = <number of intervals to collect data> \n";
  std::cerr << "\t              Tfile = <T search value, in K> \n";
  exit(1);
}

// A simple struct to hold all the parameters, experiment data and some work space
// to be accessible to the MINPACK function
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

//
// Make some static data
//
static ChemDriver cd;
Real param_eps = 1.5e-8;
MINPACKstruct mystruct(cd,param_eps);


//
// Helper functions for interfacing to MINPACK
//

//
// The real function to be minimized
//
Real 
funcF(void* p, const Array<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  Array<Real> dvals(s->expt_manager.NumExptData());
  s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  return s->parameter_manager.ComputePrior(pvals) + s->expt_manager.ComputeLikelihood(dvals);
}

//
// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
//
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

//
// Compute the derivative of the function funcF with respect to 
// the Kth variable (forward finite differences)
//
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

//
// Gradient of function to minimize, using finite differences
//
void grad(void * p, const Array<Real>& X, Array<Real>& gradF) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  for (int ii=0;ii<num_vals;ii++){
    gradF[ii] = der_ffd(p,X,ii); 
    //gradF[ii] = der_cfd(p,X,ii); 
  } 
}

//
// This is what we give to MINPACK
// It computes the gradient of the function to be minimized
//
int FCN(void       *p,    
        int	   NP,
        const Real *X,
        Real       *FVEC,
        int 	   IFLAGP)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  s->ResizeWork();
  const Array<Real> Xv(X,NP);
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

  std::cout << "Final paramters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }

  parameter_manager.ResetParametersToDefault();
  std::cout << "Reset paramters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }

  BoxLib::Finalize();
}

