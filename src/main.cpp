#include <Driver.H>
#include <ChemDriver.H>

#include <iomanip>
#include <iostream>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>

static
void 
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " pmf_file=<input fab file name> [options] \n";
  exit(1);
}

//
// The function minimized by minpack
//
static
Real
NegativeLogLikelihood(const std::vector<double>& parameters)
{
  return -Driver::LogLikelihood(parameters);
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

  Real fpIpJ = NegativeLogLikelihood(XpIpJ);
  Real fpImJ = NegativeLogLikelihood(XpImJ);
  Real fmIpJ = NegativeLogLikelihood(XmIpJ);
  Real fmImJ = NegativeLogLikelihood(XmImJ);

  return 1.0/(4.0*hI*hJ) * ( fpIpJ - fpImJ - fmIpJ + fmImJ );  
}

typedef std::vector<std::vector<Real> > MyMat;

std::pair<MyMat,MyMat>
get_INVSQRT(void *p, const std::vector<Real>& X)
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  int num_vals = str->parameter_manager.NumParams();
  MINPACKstruct::LAPACKstruct& lapack = str->lapack_struct;

  // The matrix
  std::vector<Real>& a = lapack.a;

  // Fill the upper matrix
  MyMat H(num_vals);
  for( int ii=0; ii<num_vals; ii++ ){
    H[ii].resize(num_vals);
    for( int jj=ii; jj<num_vals; jj++ ){
      a[ii + jj*num_vals] = mixed_partial_centered( p, X, ii, jj);
      H[ii][jj] = a[ii + jj*num_vals];
    }
  }

  // Do decomposition
  lapack_int info = lapack.DSYEV_wrap();
  BL_ASSERT(info == 0);

  // Get vector of eigenvalues
  const std::vector<Real>& singular_values = lapack.s;
  std::vector<Real> linv(num_vals);

  // Get 1/sqrt(lambda)
  for (int i=0; i<num_vals; ++i) {
    linv[i] = 1 / std::sqrt(singular_values[i]);
  }

  // Get invsqrt
  MyMat invsqrt(num_vals);
  for (int i=0; i<num_vals; ++i) {
    invsqrt[i].resize(num_vals,0);
    for (int j=0; j<num_vals; ++j) {
      invsqrt[i][j] = a[i + j*num_vals] * linv[j];
    }
  }
  return std::pair<MyMat,MyMat>(H,invsqrt);
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
  Real fx1 = NegativeLogLikelihood(xdX1);

  xdX2[K] -= h;
  Real fx2 = NegativeLogLikelihood(xdX2);

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

  Real fx1 = NegativeLogLikelihood(xdX);
  Real fx2 = NegativeLogLikelihood(X);

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
  std::vector<Real> Fv(NP);
  for (int i=0; i<NP; ++i) {
    Xv[i] = X[i];
  }
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


  Real Ffinal = NegativeLogLikelihood(soln);
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

#ifdef _OPENMP
#include "omp.h"
#endif

void MCSampler( void* p,
		std::vector<std::vector<Real> >& samples,
		std::vector<Real>& w,
		const std::vector<Real>& prior_mean,
		const std::vector<Real>& prior_std){
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
      //w[ii] = exp(-str->expt_manager.ComputeLikelihood(sample_data));
      w[ii] = str->expt_manager.ComputeLikelihood(sample_data);
    }
  }


#if 1
  Real wmin = w[0];
  for(int ii=1; ii<NOS; ii++){
    wmin = std::min(w[ii],wmin);
  }
  for(int ii=1; ii<NOS; ii++){
    w[ii] = std::exp(-(w[ii] - wmin));
    std::cout << "wbefore = "<< w[ii] << std::endl;
  }
#endif

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

Real F0(const std::vector<Real>& sample,
        const std::vector<Real>& mu,
        const MyMat& H,
        Real phi)
{
  int N = sample.size();
  Real F0 = 0;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      F0 += H[i][j] * (sample[j] - mu[j]) * (sample[i] - mu[i]);
    }
  }  
  return phi + 0.5*F0;
}

void SlightlyBetterSampler( void* p,
                            std::vector<std::vector<Real> >& samples,
                            std::vector<Real>& w,
                            const std::vector<Real>& mu,
                            const MyMat& H,
                            const MyMat& invsqrt,
                            Real phi)

{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  str->ResizeWork();
	  
  int num_params = str->parameter_manager.NumParams();
  int NOS = samples.size();

  std::vector<Real> sample_data(str->expt_manager.NumExptData());
  std::vector<Real> s(num_params);
  std::vector<Real> Fo(NOS);
  
  std::cout <<  " " << std::endl;
  std::cout <<  "STARTING SLIGHTLY BETTER MC SAMPLING " << std::endl;
  std::cout <<  "Number of samples: " << NOS << std::endl;

  for(int ii=0; ii<NOS; ii++){
    BL_ASSERT(samples[ii].size()==num_params);
    for(int jj=0; jj<num_params; jj++){
      samples[ii][jj] = mu[jj];
      for(int kk=0; kk<num_params; kk++){
        samples[ii][jj] += invsqrt[jj][kk]*randn();
      }
    }
    Fo[ii] = F0(samples[ii],mu,H,phi);
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
      Real F = NegativeLogLikelihood(samples[ii]);
      w[ii] = Fo[ii] - F;
      //w[ii] = exp(w[ii]);
    }
  }

#if 1
  Real wmax = w[0];
  for(int ii=1; ii<NOS; ii++){
    wmax = std::max(w[ii],wmax);
  }
  for(int ii=1; ii<NOS; ii++){
    w[ii] = std::exp(w[ii] - wmax);
    std::cout << "wbefore = "<< w[ii] << std::endl;
  }
#endif

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
  std::cout <<  "END SLIGHTLY BETTER MC SAMPLING " << std::endl;
  std::cout <<  " " << std::endl;
}


// END MATTI'S CODE

int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  // Bit to test premix wrapping
  //
  //
  ChemDriver *cd; 
  cd->SetTransport(ChemDriver::CD_TRANLIB);
  cd = new ChemDriver;

  std::cout << "Chemdriver using " << cd->numSpecies() << " species\n";
  Array<std::string> names = cd->speciesNames();
  for(int i=0; i< cd->numSpecies(); i++ ){
      std::cout << " Species " << i << " in cd is: " << names[i] << std::endl;
  }
  PREMIXReactor * pmreact;
  pmreact = new PREMIXReactor(*cd);
  pmreact->setInputFile("premix.inp_closer");
  pmreact->InitializeExperiment();
  std::vector<Real> sim_obs;
  pmreact->GetMeasurements( sim_obs);
  std::cout << "Simulated observation as: " << sim_obs[0] << std::endl;


  //
  return(0);


  Driver driver;
  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  
  const std::vector<Real>& true_data = expt_manager.TrueData();
  const std::vector<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();
  const std::vector<Real>& true_data_std = expt_manager.ObservationSTD();

  std::cout << "True and noisy data:\n"; 
  int num_data = true_data.size();
  for(int ii=0; ii<num_data; ii++){
    std::cout << "  True: " << true_data[ii]
              << "  Noisy: " << perturbed_data[ii]
              << "  Standard deviation: " << true_data_std[ii] << std::endl;
  }
   
  const std::vector<Real>& true_params = parameter_manager.TrueParameters();
  const std::vector<Real>& prior_mean = parameter_manager.PriorMean();
  const std::vector<Real>& prior_std = parameter_manager.PriorSTD();
  
  std::cout << "True and prior mean:\n"; 
  int num_params = true_params.size();
  for(int ii=0; ii<num_params; ii++){
    std::cout << "  True: " << true_params[ii]
              << "  Prior: " << prior_mean[ii]
              << "  Standard deviation: " << prior_std[ii] << std::endl;
  }

  Real Ftrue = NegativeLogLikelihood(true_params);
  std::cout << "Ftrue = " << Ftrue << std::endl;
  
  ParmParse pp;
  bool do_sample=false; pp.query("do_sample",do_sample);
  if (do_sample) {
    std::cout << "START SAMPLING"  << std::endl;
    std::vector<Real> plot_params(num_params);
    std::vector<Real> plot_data(num_data);
    std::vector<Real> plot_grad(num_params);
    std::ofstream of,of1;
    of.open("sample_data.dat");
    of1.open("sample_grad.dat");
    of << std::setprecision(20);
    of1 << std::setprecision(20);
    int Nsample = 101; pp.query("Nsample",Nsample);
    for (int i=0; i<Nsample; ++i) {
      Real eta = Real(i)/(Nsample-1);
      for(int ii=0; ii<num_params; ii++){
        plot_params[ii] = eta*true_params[ii] + (1-eta)*prior_mean[ii];
        //plot_params[ii] = eta*11963 + (1-eta)*11982;
        plot_params[ii] = eta*11975 + (1-eta)*11985;
      }

#if 0
      grad((void *)(mystruct), plot_params,plot_grad);
      Real Fplot = funcF((void*)(mystruct), plot_params);
      std::cout << eta << " " << plot_params[0] << " " << Fplot << '\n';
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



  Real F = NegativeLogLikelihood(prior_mean);
  std::cout << "F = " << F << std::endl;

#if 1
  std::cout << " starting MINPACK "<< std::endl;
  // Call minpack
  std::vector<Real> guess_params(num_params);
  std::cout << "Guess parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    guess_params[ii] = prior_mean[ii];
    std::cout << guess_params[ii] << std::endl;
  }
  std::vector<Real> guess_data(num_data);
  expt_manager.GenerateTestMeasurements(guess_params,guess_data);
  std::cout << "Guess data: " << std::endl;
  for(int ii=0; ii<num_data; ii++){
    std::cout << guess_data[ii] << std::endl;
  }

  std::vector<Real> soln_params(num_params);
  minimize((void*)(driver.mystruct), guess_params, soln_params);

  std::cout << "Final parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }

  std::vector<Real> confirm_data(num_data);
  expt_manager.GenerateTestMeasurements(soln_params,confirm_data);
  std::cout << "Confirm data: " << std::endl;
  for(int ii=0; ii<num_data; ii++){
    std::cout << confirm_data[ii] << std::endl;
  }

  Real Fconf = NegativeLogLikelihood(soln_params);
  std::cout << "Fconf = " << Fconf << std::endl;

  std::vector<Real> Gconf(num_params); 
  std::cout << "Gconf: " << std::endl;
  grad((void*)(driver.mystruct),soln_params,Gconf);
  for(int ii=0; ii<num_params; ii++){
    std::cout << Gconf[ii] << std::endl;
  }
#endif

  std::pair<MyMat,MyMat> mats = get_INVSQRT((void*)driver.mystruct, true_params);
  const MyMat& H = mats.first;
  const MyMat& invsqrt = mats.second;

// MATTI'S CODE, USE WITH EXTREME CAUTION
  int NOS = 10; pp.query("NOS",NOS);
  std::vector<Real> w(NOS);
  std::vector<std::vector<Real> > samples(NOS, std::vector<Real>(num_params,-1));
#if 1
  MCSampler((void*)(driver.mystruct),samples,w,prior_mean,prior_std);
#else
  SlightlyBetterSampler((void*)(driver.mystruct),samples,w,soln_params,H,invsqrt,Fconf);
#endif
// END MATTI'S CODE

#if 0
  parameter_manager.ResetParametersToDefault();
  std::cout << "Reset parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }
#endif
  BoxLib::Finalize();
}

