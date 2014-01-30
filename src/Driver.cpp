#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <Utility.H>
#include <ChemDriver.H>
#include <cminpack.h>
#include <stdio.h>
#include <lapacke.h>

#include <Driver.H>

#ifdef _OPENMP
#include "omp.h"
#endif

ChemDriver *Driver::cd = 0;
MINPACKstruct *Driver::mystruct = 0;
Real Driver::param_eps = 1.e-4;
Real BAD_SAMPLE_FLAG = -10000;

Real 
funcF(void* p, const std::vector<Real>& pvals)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  s->ResizeWork();

  std::pair<bool,Real> Fa = s->parameter_manager.ComputePrior(pvals);
  if (!Fa.first) return BAD_SAMPLE_FLAG;

  std::vector<Real> dvals(s->expt_manager.NumExptData());
  s->expt_manager.GenerateTestMeasurements(pvals,dvals);
  Real Fb = s->expt_manager.ComputeLikelihood(dvals);

  //std::cout << pvals[0] << " " << Fa.second + Fb << std::endl;

  return (Fa.second  +  Fb);
}

double Driver::LogLikelihood(const std::vector<double>& parameters)
{
  return -funcF((void*)(Driver::mystruct),parameters);
}

int Driver::NumParams()
{
  return Driver::mystruct->parameter_manager.NumParams();
}


Real mixed_partial_centered (void* p, const std::vector<Real>& X, int i, int j)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  BL_ASSERT(s->work_array_len >= X.size());
  std::vector<Real>& XpIpJ = s->Work(0);
  std::vector<Real>& XmIpJ = s->Work(1);
  std::vector<Real>& XpImJ = s->Work(2);
  std::vector<Real>& XmImJ = s->Work(3);

  int num_vals = s->parameter_manager.NumParams();

  for (int ii=0; ii<num_vals; ii++){
   XpIpJ [ii] = X[ii];
   XpImJ [ii] = X[ii];
   XmIpJ [ii] = X[ii];
   XmImJ [ii] = X[ii];
  }
                
  Real typI = std::max(s->parameter_manager.TypicalValue(i), std::abs(X[i]));
  Real typJ = std::max(s->parameter_manager.TypicalValue(j), std::abs(X[j]));

  Real hI = typI * s->param_eps;
  Real hJ = typJ * s->param_eps;

  XpIpJ[i] += hI;
  XpIpJ[j] += hJ;

  XpImJ[i] += hI;
  XpImJ[j] -= hJ;

  XmIpJ[i] -= hI;
  XmIpJ[j] += hJ;

  XmImJ[i] -= hI;
  XmImJ[j] -= hJ;

  Real fpIpJ = funcF(p, XpIpJ);
  Real fpImJ = funcF(p, XpImJ);
  Real fmIpJ = funcF(p, XmIpJ);
  Real fmImJ = funcF(p, XmImJ);

  return 1.0/(4.0*hI) * ( fpIpJ - fpImJ - fmIpJ + fmImJ );
  
}

/*
 * Load Hessian at X, perform SVD, results stored internally (RG, MSD 2014)
 */
void
get_Hessian_SVD(void *p, const std::vector<Real>& X)
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  int num_vals = str->parameter_manager.NumParams();
  MINPACKstruct::LAPACKstruct& lapack = str->lapack_struct;
  std::vector<Real>& a = lapack.a;
  for( int jj=0; jj<num_vals; jj++ ){
    for( int ii=0; ii<num_vals; ii++ ){
      a[ii + jj*num_vals] = mixed_partial_centered( p, X, ii, jj);
    }
  }
  lapack_int info = lapack.DGESVD_wrap();
  BL_ASSERT(info == 0);
}

//
// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
//
Real
der_cfd(void* p, const std::vector<Real>& X, int K) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  std::vector<Real>& xdX1 = s->Work(0);
  std::vector<Real>& xdX2 = s->Work(1);

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
der_ffd(void* p, const std::vector<Real>& X, int K) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  std::vector<Real>& xdX = s->Work(0);

  int num_vals = s->parameter_manager.NumParams();

  for (int ii=0; ii<num_vals; ii++){
    xdX[ii]  = X[ii];
  }
                
  Real typ = std::max(s->parameter_manager.TypicalValue(K), std::abs(xdX[K]));
  Real h = typ * s->param_eps;

  xdX[K] += h;

  Real fx1 = funcF(p, xdX);
  Real fx2 = funcF(p, X);

  return (fx1-fx2)/(xdX[K]-X[K]);
}

//
// Gradient of function to minimize, using finite differences
//
void grad(void * p, const std::vector<Real>& X, std::vector<Real>& gradF) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  for (int ii=0;ii<num_vals;ii++){
    //gradF[ii] = der_ffd(p,X,ii); 
    gradF[ii] = der_cfd(p,X,ii);
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
  std::vector<Real> Xv(NP);
  for (int i=0; i<NP; ++i) {
    Xv[i] = X[i];
  }
  std::vector<Real> Fv(NP);
  grad(p,Xv,Fv);
  for (int i=0; i<NP; ++i) {
    FVEC[i] = Fv[i];
  }
  if (IFLAGP==1) {
    std::cout << "parameter, grad = " << X[0] << ", " << FVEC[0] << std::endl; 
  }
  return 0;
}

// Call minpack
void minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  std::vector<Real> FVEC(num_vals);
  int INFO;

  int MAXFEV=1e8,ML=num_vals-1,MU=num_vals-1,NPRINT=1,LDFJAC=num_vals;
  int NFEV;
  int LR = 0.5*(num_vals*(num_vals+1)) + 1;
  std::vector<Real> R(LR);
  std::vector<Real> QTF(num_vals);
  std::vector<Real> DIAG(num_vals);

  int MODE = 2;
  if (MODE==2) {
    for (int i=0; i<num_vals; ++i) {
      DIAG[i] = std::abs(s->parameter_manager[i].DefaultValue());
    }
  }
  Real EPSFCN=1e-6;
  std::vector<Real> FJAC(num_vals*num_vals);

  Real XTOL=1.e-6;
  Real FACTOR=100;
  std::vector< std::vector<Real> > WA(4, std::vector<Real>(num_vals));

  soln = guess;
  INFO = hybrd(FCN,p,num_vals,&(soln[0]),&(FVEC[0]),XTOL,MAXFEV,ML,MU,EPSFCN,&(DIAG[0]),
               MODE,FACTOR,NPRINT,&NFEV,&(FJAC[0]),LDFJAC,&(R[0]),LR,&(QTF[0]),
               &(WA[0][0]),&(WA[1][0]),&(WA[2][0]),&(WA[3][0]));   

  std::cout << "minpack INFO: " << INFO << std::endl;
  if(INFO==0)
  {
	std::cout << "minpack: improper input parameters " << std::endl;
  }
  else if(INFO==1)
  {
	std::cout << "minpack: relative error between two consecutive iterates is at most XTOL" << std::endl;
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
  FCN(p,num_vals,&(soln[0]),&(FVEC[0]),IFLAGP);
  std::cout << "X, FVEC: " << std::endl;
  for(int ii=0; ii<num_vals; ii++){
    std::cout << soln[ii] << " " << FVEC[ii] << std::endl;
  }
};

// MATTI'S CODE, USE WITH EXTREME CAUTION

void NormalizeWeights(std::vector<Real>& w){
  int NOS = w.size();
  Real SumWeights = 0;
  for(int ii=0; ii<NOS; ii++){
	SumWeights = SumWeights+w[ii];
  }
  for(int ii=0; ii<NOS; ii++){
	w[ii] = w[ii]/SumWeights;
  }
}

Real EffSampleSize(std::vector<Real>& w, int NOS){
   // Approximate effective sample size
   Real SumSquaredWeights = 0;
   for(int ii=0; ii<NOS; ii++){
	   SumSquaredWeights = SumSquaredWeights + w[ii]*w[ii];
   }
   Real Neff = 1/SumSquaredWeights; 
   return Neff;
}


void Mean(std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples){
  int NOS = samples.size();
  int num_params = samples[1].size();
  for(int ii=0; ii<num_params; ii++){
	  Mean[ii] = 0; // initialize  
	  for(int jj=0; jj<NOS; jj++){
		  Mean[ii] = Mean[ii]+samples[jj][ii];
	  }
	  Mean[ii] = Mean[ii]/(Real)NOS;
  }	
}


void WeightedMean(std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples){
  int NOS = samples.size();
  int num_params = samples[1].size();
  for(int ii=0; ii<num_params; ii++){
	  CondMean[ii] = 0; // initialize  
	  for(int jj=0; jj<NOS; jj++){
		  CondMean[ii] = CondMean[ii]+w[jj]*samples[jj][ii];
	  }
  }	
}


void Var(std::vector<Real>& Var,std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples){
  int NOS = samples.size();
  int num_params = samples[1].size();
  for(int ii=0; ii<num_params; ii++){	  
  Var[ii] = 0;
  for(int jj=0; jj<NOS; jj++){
	  Var[ii] = Var[ii] + (samples[jj][ii]-Mean[ii])*(samples[jj][ii]-Mean[ii]);
  }
  Var[ii] = Var[ii]/NOS;
  }
}


void WeightedVar(std::vector<Real>& CondVar,std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples){
  int NOS = samples.size();
  int num_params = samples[1].size();	
  for(int ii=0; ii<num_params; ii++){	  
  CondVar[ii] = 0;
  for(int jj=0; jj<NOS; jj++){
	  CondVar[ii] = CondVar[ii] + w[jj]*(samples[jj][ii]-CondMean[ii])*(samples[jj][ii]-CondMean[ii]);
  }
  }
}


void WriteSamplesWeights(std::vector<std::vector<Real> >& samples, std::vector<Real>& w){
  int NOS = samples.size();
  int num_params = samples[1].size();
  std::ofstream of,of1;
  of.open("samples.dat");
  of1.open("weights.dat");
  of << std::setprecision(20);
  of1 << std::setprecision(20);
  for(int ii=0;ii<NOS;ii++){
      of1 << w[ii] << '\n';
  }
  for(int jj=0;jj<NOS;jj++){
  	for(int ii=0;ii<num_params;ii++){
		  of << samples[jj][ii] << " ";
	  }
	of << '\n';
  }
  of.close();
  of1.close();
}


void Resampling(std::vector<std::vector<Real> >& Xrs,std::vector<Real>& w,std::vector<std::vector<Real> >& samples){
  int NOS = samples.size();
  int num_params = samples[1].size();
  std::vector<Real> c(NOS+1);
  for(int jj = 0;jj < NOS+1; jj++){
	 c[jj] = 0; // initialize	  
  }
  // construct cdf
  for(int jj=1;jj<NOS+1;jj++){
	  c[jj]=c[jj-1]+w[jj-1];
  }
  // sample it and get the stronger particles more often
  int ii = 0; // initialize
  Real u1 = drand()/NOS;
  Real u = 0;
  for(int jj=0;jj<NOS;jj++){
    u = u1+(Real)jj/NOS;
    while(u>c[ii]){
        ii++;
    }
    for(int kk=0;kk<num_params;kk++){
	    Xrs[jj][kk] = samples[(ii-1)][kk];
    }
  }
}


void WriteResampledSamples(std::vector<std::vector<Real> >& Xrs){
  int NOS = Xrs.size();
  int num_params = Xrs[1].size();
  std::ofstream of2;
  of2.open("resampled_samples.dat");
  of2 << std::setprecision(20);
  for(int jj=0;jj<NOS;jj++){
	  for(int ii=0;ii<num_params;ii++){
		  of2 << Xrs[jj][ii] << " ";
	  }
	  of2 << '\n';
  }
  of2.close();
}

void MCSampler( void* p,
		std::vector<std::vector<Real> >& samples,
		std::vector<Real>& w,
		std::vector<Real>& prior_mean,
		std::vector<Real>& prior_std){
  MINPACKstruct *str = (MINPACKstruct*)(p);
  str->ResizeWork();
	  
  int num_params = str->parameter_manager.NumParams();
  int NOS = samples.size();

  std::vector<Real> sample_data(str->expt_manager.NumExptData());
  std::vector<Real> s(num_params);
  
  std::cout <<  " " << std::endl;
  std::cout <<  "STARTING BRUTE FORCE MC SAMPLING " << std::endl;
  std::cout <<  "Number of samples: " << NOS << std::endl;
  for(int ii=0; ii<NOS; ii++){
    BL_ASSERT(samples[ii].size()==num_params);
    for(int jj=0; jj<num_params; jj++){
      samples[ii][jj] = prior_mean[jj] + prior_std[jj]*randn();
    }
  }

#ifdef _OPENMP
  int tnum = omp_get_max_threads();
  BL_ASSERT(tnum>0);
  std::cout <<  " number of threads: " << tnum << std::endl;
#else
  int tnum = 1;
#endif
  Real dthread = NOS / Real(tnum);
  std::vector<int> trange(tnum);
  for (int ithread = 0; ithread < tnum; ithread++) {
    trange[ithread] = ithread * dthread;
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int ithread = 0; ithread < tnum; ithread++) {
    int iBegin = trange[ithread];
    int iEnd = (ithread==tnum-1 ? NOS : trange[ithread+1]);

    for(int ii=iBegin; ii<iEnd; ii++){
      str->expt_manager.GenerateTestMeasurements(samples[ii],sample_data);
      w[ii] = exp(-str->expt_manager.ComputeLikelihood(sample_data));
    }
  }

  // Normalize weights, print to terminal
  NormalizeWeights(w);
  for(int ii=0; ii<NOS; ii++){
    for(int jj=0; jj<num_params; jj++){
      std::cout << "Sample " << samples[ii][jj] <<  " weight = "<< w[ii] << std::endl;
    }
  }
  
  // Approximate effective sample size	
  Real Neff = EffSampleSize(w,NOS);
  std::cout <<  " " << std::endl;
  std::cout <<  "Effective sample size = "<< Neff << std::endl;

  // Compute conditional mean
  std::vector<Real> CondMean(num_params);
  WeightedMean(CondMean, w, samples);

  // Variance
  std::vector<Real> CondVar(num_params);
  WeightedVar(CondVar,CondMean, w, samples);

  // Print stuff to screen
  for(int jj=0; jj<num_params; jj++){
	  std::cout <<  "Prior mean = "<< prior_mean[jj] << std::endl;
	  std::cout <<  "Conditional mean = "<< CondMean[jj] << std::endl;
	  std::cout <<  "Standard deviation = "<< sqrt(CondVar[jj]) << std::endl;
  }

  // Write samples and weights into files
  WriteSamplesWeights(samples, w);

  // Resampling
  std::vector<std::vector<Real> > Xrs(NOS, std::vector<Real>(num_params,-1));// resampled parameters
  Resampling(Xrs,w,samples);
  WriteResampledSamples(Xrs);

  // Compute conditional mean after resampling
  std::vector<Real> CondMeanRs(num_params);
  Mean(CondMeanRs, Xrs);
  
  // Variance after resampling
  std::vector<Real> CondVarRs(num_params);
  Var(CondVarRs, CondMeanRs, Xrs);

  // Print results of resampling
  for(int jj=0; jj<num_params; jj++){
	  std::cout <<  "Conditional mean after resampling = "<< CondMeanRs[jj] << std::endl;
	  std::cout <<  "Standard deviation after resampling = "<< sqrt(CondVarRs[jj]) << std::endl;
  }


  std::cout <<  " " << std::endl;
  std::cout <<  "END BRUTE FORCE MC SAMPLING " << std::endl;
  std::cout <<  " " << std::endl;
}
// END MATTI'S CODE

Driver::Driver()
{
  cd = new ChemDriver;
  ParmParse pp("driver");
  param_eps = 1.e-4; pp.query("param_eps",param_eps);
  mystruct = new MINPACKstruct(*cd,param_eps);

  ParameterManager& parameter_manager = mystruct->parameter_manager;
  ExperimentManager& expt_manager = mystruct->expt_manager;  
  
  CVReactor *cv_reactor = new CVReactor(*cd);
  expt_manager.AddExperiment(cv_reactor,"exp1");
  expt_manager.InitializeExperiments();

  parameter_manager.Clear();
  std::vector<Real> true_params;
  // Reactions that seem to matter: 0, 15, 41, 49, 135, 137, 155 (15, 135 strongest)
  true_params.push_back(parameter_manager.AddParameter(13,ChemDriver::FWD_EA));
  int num_params = parameter_manager.NumParams();

  int num_data = expt_manager.NumExptData();
  std::vector<Real> true_data(num_data);
    
  expt_manager.GenerateTestMeasurements(true_params,true_data);

  std::vector<Real> true_data_std(num_data);
  for(int ii=0; ii<num_data; ii++){
    true_data_std[ii] = 15;
  }
  expt_manager.InitializeTrueData(true_data,true_data_std);

  expt_manager.GenerateExptData(); // Create perturbed experimental data (stored internally)
  const std::vector<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();
   
  std::vector<Real> lower_bound(num_params);
  std::vector<Real> upper_bound(num_params);
  std::vector<Real> prior_mean(num_params);
  std::vector<Real> prior_std(num_params);
  for(int ii=0; ii<num_params; ii++){
    prior_mean[ii] = 11976;
    prior_std[ii] = 1000;
    lower_bound[ii] = 1000;
    upper_bound[ii] = 20000;
  }

  parameter_manager.SetStatsForPrior(prior_mean,prior_std,lower_bound,upper_bound);
}

Driver::~Driver()
{
  delete mystruct;
  delete cd;
}
