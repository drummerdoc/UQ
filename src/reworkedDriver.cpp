#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <Utility.H>
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

static
Real
get_macheps() {
  Real h = 1;
  while (1+h != 1) {
    h *= 0.5;
  }
  Real mach_eps = std::sqrt(h);
  std::cout << "Setting macheps to " << mach_eps << std::endl;
  return mach_eps;
}

static ChemDriver* cd;
MINPACKstruct *mystruct;


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
  s->ResizeWork();
  Array<Real> dvals(s->expt_manager.NumExptData());
  s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  Real Fa = s->parameter_manager.ComputePrior(pvals);
  Real Fb = s->expt_manager.ComputeLikelihood(dvals);

  //std::cout << "F, Fa, Fb: " << Fa + Fb << ", " << Fa << ", " << Fb << std::endl;
  //std::cout << "Parameter: " << pvals[0] << std::endl;
  
  return Fa  +  Fb;
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

  // std::cout << "typ from p: " << s->parameter_manager.TypicalValue(K) << std::endl;
  // std::cout << "typ: " << typ << std::endl;
  // std::cout << "h: " << h << std::endl;
  // std::cout << "X: " << X[0] << std::endl;
  // std::cout << "xdX: " << xdX[0] << std::endl;
  // std::cout << "fx1: " << fx1 << std::endl;
  // std::cout << "fx2: " << fx2 << std::endl;
  // std::cout << "df: " << fx1-fx2 << std::endl;
  // std::cout << "dx: " << xdX[K]-X[K] << std::endl;
  // std::cout << "der: " << der << std::endl;
  // BoxLib::Abort();

  return der;
}

//
// Gradient of function to minimize, using finite differences
//
void grad(void * p, const Array<Real>& X, Array<Real>& gradF) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  for (int ii=0;ii<num_vals;ii++){
    //gradF[ii] = der_ffd(p,X,ii); 
    gradF[ii] = der_cfd(p,X,ii);
    std::cout << "parameter = " << X[0] << std::endl; 
    std::cout << "grad = " << gradF[0] << std::endl; 
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
  Array<Real> Fv(NP);
  grad(p,Xv,Fv);
  for (int i=0; i<NP; ++i) {
    FVEC[i] = Fv[i];
  }
  return 0;
}

// Call minpack
void minimize(void *p, const Array<Real>& guess, Array<Real>& soln)
{
#if 1
  int INFO,LWA=180;
  Real TOL=1.5e-14;
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  Array<Real> FVEC(num_vals);
  Array<Real> WA(LWA);
  soln = guess;
  INFO = hybrd1(FCN,p,num_vals,soln.dataPtr(),FVEC.dataPtr(),TOL,WA.dataPtr(),WA.size());   
  std::cout << "minpack INFO: " << INFO << std::endl;
  if(INFO==1)
  {
	std::cout << "minpack: improper input parameters " << std::endl;
  }
  else if(INFO==2)
  {
	std::cout << "minpack: algorithm estimates that the relative error between X and the solution is at most TOL " << std::endl;
  }
   else if(INFO==3)
  {
	std::cout << "minpack: number of calls to FCN has reached or exceeded 200*(N+1)" << std::endl;
  }
  else if(INFO==4)
  {
  	std::cout << "minpack: iteration is not making good progress" << std::endl;
  }
#endif

#if 0
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  Array<Real> FVEC(num_vals);


  int INFO,LWA=180,MAXFEV=1e8,ML=1,MU=1,EPSFCN=1e-4,DIAG=1,MODE=1,NPRINT=0,LDFJAC=1;
  int NFEV;
  int LR = 1;
  Array<Real> R(LR);
  Array<Real> QTF(num_vals);

  Real XTOL=1.e-4;
  Real FACTOR=100;
  Array<Real> WA1(num_vals);
  Array<Real> WA2(num_vals);
  Array<Real> WA3(num_vals);
  Array<Real> WA4(num_vals);

  soln = guess;
  hybrd(FCN,p,num_vals,soln.dataPtr(),FVEC.dataPtr(),XTOL,MAXFEV,ML,MU,EPSFCN,DIAG,MODE,FACTOR,NPRINT,INFO,NFEV,LDFJAC,R,LR,QTF,WA1,WA2,WA3,WA4);   

  std::cout << "minpack INFO: " << INFO << std::endl;
  if(INFO==0)
  {
	std::cout << "minpack: improper input parameters " << std::endl;
  }
  else if(INFO==1)
  {
	std::cout << "minpack: elative error between two consecutive iterates is at most XTOL" << std::endl;
  }
  else if(INFO==2)
  {
	std::cout << "minpack: number of calls to FCN has reached or exceeded MAXFEV" << std::endl;
  }
   else if(INFO==3)
  {
	std::cout << "minpack: XTOL is too small.  No further improvement in the approximate solution X is possible." << std::endl;
  }
  else if(INFO==4)
  {
  	std::cout << "minpack: iteration is not making good progress, as measured by the improvement from the last five Jacobian evaluations."<< std::endl;
  }
  else if(INFO==5)
  {
  	std::cout << "minpack: iteration is not making good progress, as measured by the improvement from the last ten iterations. "<< std::endl;
  }


  Real Ffinal = funcF(p,soln);
  std::cout << "Ffinal: " << Ffinal << std::endl;

  int IFLAGP = 0;
  int ret = FCN(p,num_vals,soln.dataPtr(),FVEC.dataPtr(),IFLAGP);
  std::cout << "X, FVEC: " << std::endl;
  for(int ii=0; ii<num_vals; ii++){
    std::cout << soln[ii] << " " << FVEC[ii] << std::endl;
  }
#endif
};

int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  if (argc<2) print_usage(argc,argv);

  cd = new ChemDriver;

  Real eps = get_macheps();
  Real param_eps = 1.e-4;
  mystruct = new MINPACKstruct(*cd,param_eps);

  ParameterManager& parameter_manager = mystruct->parameter_manager;
  ExperimentManager& expt_manager = mystruct->expt_manager;  
  

  Array<Real> true_params;
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_BETA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::FWD_EA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_A));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_BETA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::LOW_EA));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_A));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TS));
  // true_params.push_back(parameter_manager.AddParameter(8,ChemDriver::TROE_TSSS));

  //true_params.push_back(parameter_manager.AddParameter(9,ChemDriver::FWD_A));
  true_params.push_back(parameter_manager.AddParameter(0,ChemDriver::FWD_EA));
  int num_params = parameter_manager.NumParams();
  std::cout << "NumParams:" << num_params << std::endl; 
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  True: " << parameter_manager[ii] << std::endl;
  }

  CVReactor cv_reactor(*cd);
  CVReactor cv_reactor2(*cd);
  expt_manager.AddExperiment(cv_reactor,"exp1");
  //expt_manager.AddExperiment(cv_reactor2,"exp2");
  expt_manager.InitializeExperiments();

  int num_data = expt_manager.NumExptData();
  std::cout << "NumData:" << num_data << std::endl; 
  Array<Real> true_data(num_data);
  Array<Real> true_data_std(num_data,.01); // Set variance of data 

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
    prior_std[ii] = 6000;
    if (prior_std[ii] == 0) {prior_std[ii] = 1e-2;}
    prior_mean[ii] = 10000;
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

  Real Ftrue = funcF((void*)(mystruct),true_params);
  std::cout << "Ftrue = " << Ftrue << std::endl;


 
  ParmParse pp;
  bool do_sample=false; pp.query("do_sample",do_sample);
  std::cout << "START SAMPLING"  << std::endl;
  if (do_sample) {
    Array<Real> plot_params(num_params);
    Array<Real> plot_data(num_data);
    Array<Real> plot_grad(num_params);
    std::ofstream of,of1;
    of.open("sample_data.dat");
    of1.open("sample_grad.dat");
    of << std::setprecision(20);
    of1 << std::setprecision(20);
    int Nsample = 101; pp.query("Nsample",Nsample);
    for (int i=0; i<Nsample; ++i) {
      Real eta = Real(i)/(Nsample-1);
      for(int ii=0; ii<num_params; ii++){
        //plot_params[ii] = eta*true_params[ii] + (1-eta)*prior_mean[ii];
        //plot_params[ii] = 0.98*eta*true_params[ii] + (1-eta)*prior_mean[ii]*1.01;
	plot_params[ii] = eta*10000 + (1-eta)*17000;
      }

#if 1
      grad((void *)(mystruct), plot_params,plot_grad);
      Real Fplot = funcF((void*)(mystruct), plot_params);
      of << plot_params[0] << " " << Fplot << '\n';
      of1 << plot_params[0] << " " <<plot_grad[0] << '\n';
#endif

#if 0
      expt_manager.GenerateTestMeasurements(plot_params,plot_data);
      
      of << "VARIABLES = TIME VAL\n";
      of << BoxLib::Concatenate("ZONE T=\"",i,5) << "\"\n";
      of << "I = " << num_data << " ZONETYPE=Ordered DATAPACKING=POINT\n";
      for (int ii=0; ii<num_data; ++ii) {
        of << times[ii] << " " << plot_data[ii] << std::endl;
      }
#endif
    }
    of.close();
    of1.close();
    exit(0);
  }



  Real F = funcF((void*)(mystruct), prior_mean);  	
  std::cout << "F = " << F << std::endl;

#if 1
  std::cout << " starting MINPACK "<< std::endl;
  // Call minpack
  Array<Real> guess_params(num_params);
  std::cout << "Guess parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    //guess_params[ii] = prior_mean[ii];
    guess_params[ii] = 10000;
    std::cout << guess_params[ii] << std::endl;
  }
  Array<Real> guess_data(num_data);
  expt_manager.GenerateTestMeasurements(guess_params,guess_data);
  std::cout << "Guess data: " << std::endl;
  for(int ii=0; ii<num_data; ii++){
    std::cout << guess_data[ii] << std::endl;
  }

  Array<Real> soln_params(num_params);
  minimize((void*)(mystruct), guess_params, soln_params);

  std::cout << "Final parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }

  Array<Real> confirm_data(num_data);
  expt_manager.GenerateTestMeasurements(soln_params,confirm_data);
  std::cout << "Confirm data: " << std::endl;
  for(int ii=0; ii<num_data; ii++){
    std::cout << confirm_data[ii] << std::endl;
  }

  Real Fconf = funcF((void*)(mystruct),soln_params);
  std::cout << "Fconf = " << Fconf << std::endl;

  Array<Real> Gconf(num_params); 
  std::cout << "Gconf: " << std::endl;
  grad((void*)(mystruct),soln_params,Gconf);
  for(int ii=0; ii<num_params; ii++){
    std::cout << Gconf[ii] << std::endl;
  }

  parameter_manager.ResetParametersToDefault();
  std::cout << "Reset parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }
# endif
  delete mystruct;
  delete cd;
  BoxLib::Finalize();
}
