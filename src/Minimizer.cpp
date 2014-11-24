#include <Minimizer.H>
#include <Driver.H>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

#include <ParmParse.H>

#ifdef _OPENMP
#include "omp.h"
#endif

static real sqrt2Inv = 1/std::sqrt(2);
static int GOOD_EVAL_FLAG = 0;
static int BAD_DATA_FLAG = 1;
static int BAD_EXPT_FLAG = 2;

#define CEN_DIFF
#undef FWD_DIFF


Real
NegativeLogLikelihood(const std::vector<double>& parameters)
{
  return -Driver::LogLikelihood(parameters);
}


// /////////////////////////////////////////////////////////
// Centered differences for Hessian
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static
Real mixed_partial_centered (void* p, const std::vector<Real>& X, int i, int j)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);

  Real typI = std::max(s->parameter_manager.TypicalValue(i), std::abs(X[i]));
  Real typJ = std::max(s->parameter_manager.TypicalValue(j), std::abs(X[j]));

  Real hI = typI * s->param_eps * 10;
  Real hJ = typJ * s->param_eps * 10;

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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// Compute Hessian with finite differences
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
MyMat
Minimizer::FD_Hessian(void *p, const std::vector<Real>& X)
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  int n = str->parameter_manager.NumParams();

  // Fill the upper matrix
  MyMat H(n);
  for( int ii=0; ii<n; ii++ ){
    H[ii].resize(n);
    for (int j=0; j<n; ++j) {
      H[ii][j] = -1;
    }
    for( int jj=ii; jj<n; jj++ ){
      H[ii][jj] = mixed_partial_centered( p, X, ii, jj);
    }
  }

  for( int ii=0; ii<n; ii++ ){
    for( int jj=ii; jj<n; jj++ ){
      H[jj][ii] = H[ii][jj];
    }
  }

  return H;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// Inverse of a square root of a matrix
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
MyMat
Minimizer::InvSqrt(void *p, const MyMat & H)
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  int num_vals = str->parameter_manager.NumParams();
  MINPACKstruct::LAPACKstruct& lapack = str->lapack_struct;

  // For Lapack 
  std::vector<Real>& a = lapack.a;

  // Convert MyMat to Lapack readable
  for( int ii=0; ii<num_vals; ii++ ){
    for (int j=0; j<num_vals; ++j) {
      a[j + ii*num_vals] = -1;
    }
    for( int jj=ii; jj<num_vals; jj++ ){
	    a[jj + ii*num_vals] = H[ii][jj];
    }
  }

  // Eigenvalue decomposition
  lapack_int info = lapack.DSYEV_wrap();
  BL_ASSERT(info == 0);

  // Get vector of eigenvalues
  const std::vector<Real>& eigenvalues = lapack.s;
  
  // Display eigenvalues
  for (int i=0; i<num_vals; ++i) {
	  std::cout <<  "Eigenvalue: " << eigenvalues[i] << std::endl;
  }

  std::vector<Real> typEig(num_vals,0);
  const std::vector<Real>& priorSTD = str->parameter_manager.PriorSTD();
  for (int i=0; i<num_vals; ++i) {
    typEig[i] = 1/(priorSTD[i] * priorSTD[i]);
  }
  Real minEig = std::abs(typEig[0]);
  for (int i=1; i<num_vals; ++i) {
    minEig = std::min(minEig,std::abs(typEig[i]));
  }
  Real CutOff = 0.1 * minEig;

  // Get 1/sqrt(lambda)
  std::vector<Real> sqrtlinv(num_vals);
  for (int i=0; i<num_vals; ++i) {
    if(eigenvalues[i] < CutOff){
      sqrtlinv[i] = 0;
    }
    else{
      sqrtlinv[i] = 1 / std::sqrt(eigenvalues[i]);
    }
  }

  // Assemble inverse of square root
  MyMat invsqrt(num_vals);
  for (int i=0; i<num_vals; ++i) {
    invsqrt[i].resize(num_vals,0);
    for (int j=0; j<num_vals; ++j) {
      invsqrt[i][j] = a[j + i*num_vals] * sqrtlinv[j];
    }
  }
  return invsqrt;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




#ifdef CEN_DIFF
// /////////////////////////////////////////////////////////
// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static Real
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
#endif

#ifdef FWD_DIFF
// /////////////////////////////////////////////////////////
// Compute the derivative of the function funcF with respect to 
// the Kth variable (forward finite differences)
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static Real
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
#endif


// /////////////////////////////////////////////////////////
// Gradient of function to minimize, using finite differences
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static void grad(void * p, const std::vector<Real>& X, std::vector<Real>& gradF) {
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int num_vals = s->parameter_manager.NumParams();
  for (int ii=0;ii<num_vals;ii++){
#ifdef FWD_DIFF
    gradF[ii] = der_ffd(p,X,ii); 
#endif
#ifdef CEN_DIFF
    gradF[ii] = der_cfd(p,X,ii);
#endif
  } 
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// This is what we give to MINPACK's routine for nonlinear 
// equations (we avoid this now).
// It computes the gradient of the function to be minimized
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static
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
    std::cout << "parameter = { ";
    for (int i=0; i<NP; ++i) {
      std::cout << X[i] << " ";
    }
    std::cout << "}, grad =  {";
    for (int i=0; i<NP; ++i) {
      std::cout << FVEC[i] << " ";
    }
    std::cout << "}\n";
  }
  return 0;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////



// /////////////////////////////////////////////////////////
// Part of the function evaluated within nonlinear least squares
// This is the part that comes from the data and is computed
// with finite differences
// // /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static int eval_nlls_data(void *p, const std::vector<Real>& pvals, real *fvec)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  ExperimentManager& em = s->expt_manager;
  const std::vector<Real>& observation_std = em.ObservationSTD();
  const std::vector<Real>& perturbed_data = em.TrueDataWithObservationNoise();
  std::vector<Real> dvals(em.NumExptData());
  bool expts_ok = s->expt_manager.GenerateTestMeasurements(pvals,dvals);

  if (!expts_ok) { // Bad experiment
    return BAD_EXPT_FLAG;
  }

  for (int i=0; i<em.NumExptData(); ++i) {
    fvec[i] = sqrt2Inv * (perturbed_data[i] - dvals[i]) / observation_std[i];
  }

  return GOOD_EVAL_FLAG;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////

// /////////////////////////////////////////////////////////
// Functions of the nonlinear least squares problem
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static int eval_nlls_funcs(void *p, int m, int n, const real *x, real *fvec)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  ParameterManager& pm = s->parameter_manager;
  ExperimentManager& em = s->expt_manager;
  BL_ASSERT(n == pm.NumParams());
  int nd = m - n;
  BL_ASSERT(nd == em.NumExptData());

  const std::vector<Real>& prior_mean = pm.PriorMean();
  const std::vector<Real>& prior_std = pm.PriorSTD();
  const std::vector<Real>& upper_bound = pm.UpperBound();
  const std::vector<Real>& lower_bound = pm.LowerBound();
  std::vector<Real> pvals(n);

  bool sample_oob = false;
  for (int i=0; i<n && !sample_oob; ++i) {
    pvals[i] = x[i];
    sample_oob |= (pvals[i] < lower_bound[i] || pvals[i] > upper_bound[i]);
    fvec[i] = sqrt2Inv * (prior_mean[i] - pvals[i]) / prior_std[i];
  }

  if (sample_oob) { // Bad data
    return BAD_DATA_FLAG;
  }

  return eval_nlls_data(p,pvals,&(fvec[n]));
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




// /////////////////////////////////////////////////////////
// Nonlinear last squares function we pass to Minpack
//
// MARK PLEASE CHECK/CLEAN HERE
//
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static
int NLLSFCN(void *p, int m, int n, const real *x, real *fvec, real *fjac, 
            int ldfjac, int iflag)
{
  if (iflag == 0) {
    std::cout << "NLLSFCN status X: { ";
    for (int i=0; i<n; ++i) {
      std::cout << x[i] << " ";
    }
    std::cout << "} ";

    Real sum = 0;
    for (int i=0; i<m; ++i) {
      sum += fvec[i]*fvec[i];
    }
    std::cout << " F = " << sum << std::endl;
  }
  else if (iflag == 1) { // Evaluate functions only, do not touch FJAC
    int eflag = eval_nlls_funcs(p,m,n,x,fvec);
    if (eflag != GOOD_EVAL_FLAG) {
      if (eflag == BAD_DATA_FLAG) {
        MINPACKstruct *s = (MINPACKstruct*)(p);
        ParameterManager& pm = s->parameter_manager;
        const std::vector<Real>& upper_bound = pm.UpperBound();
        const std::vector<Real>& lower_bound = pm.LowerBound();
        std::cout << "Bad parameters" << std::endl;
        for (int i=0; i<n; ++i) {
          if (x[i] < lower_bound[i] || x[i] > upper_bound[i]) {
            std::cout << "    parameter " << i << " oob: " << x[i] << " " << lower_bound[i] << " " << upper_bound[i] << std::endl;
          }
        }
      }
      BoxLib::Abort("HANDLE BAD EVAL FLAG");
    }
  }
  else if (iflag == 2) { // Evaluate jacobian only, do not touch FVEC

    MINPACKstruct *s = (MINPACKstruct*)(p);
    ParameterManager& pm = s->parameter_manager;
    ExperimentManager& em = s->expt_manager;
    BL_ASSERT(n == pm.NumParams());
    const std::vector<Real>& prior_std = pm.PriorSTD();
    const std::vector<Real>& upper_bound = pm.UpperBound();
    const std::vector<Real>& lower_bound = pm.LowerBound();

    std::vector<Real> pvals(n);
    bool sample_oob = false;
    for (int i=0; i<n && !sample_oob; ++i) {
      pvals[i] = x[i];
      sample_oob |= (pvals[i] < lower_bound[i] || pvals[i] > upper_bound[i]);
    }
    if (sample_oob) { // Bad data
      BoxLib::Abort("Bad sample data");
    }

    std::vector<Real> fptmp(em.NumExptData());
    std::vector<Real> fmtmp(em.NumExptData());
    for (int i=0; i<n; ++i) {
      for (int j=0; j<n; ++j) {
        //fjac[n*i + j] = 0; // Row major
        fjac[m*i + j] = 0; // column major
      }
      //fjac[(n+1)*i] = - sqrt2Inv / prior_std[i]; // Row major
      fjac[(m+1)*i] = - sqrt2Inv / prior_std[i]; // Column major
    }


    int nd = m - n;


#if 0
    std::vector<Real> f0tmp(em.NumExptData());
    int ret0 = eval_nlls_data(p,pvals,&(f0tmp[0]));
    BL_ASSERT(ret0 == GOOD_EVAL_FLAG);

    for (int i=0; i<n; ++i) {
        
      std::vector<bool> this_one_good(nd);
      for (int j=0; j<nd; ++j) {
        this_one_good[j] = false;
      }
      Array<Real> val_on_first_pass(nd);
      Array<int> num_reqd(nd,0);
      int NiterMAX = 50;
      bool more_work = true;
      bool done;

      for (int Niter=0; Niter<NiterMAX && more_work; ++Niter) {

        Real typ = std::max(s->parameter_manager.TypicalValue(i), std::abs(pvals[i]));
        Real h = typ * s->param_eps * std::pow(3,Niter);
        pvals[i] += h;
        
        int ret = eval_nlls_data(p,pvals,&(fptmp[0]));
        BL_ASSERT(ret == GOOD_EVAL_FLAG);
        
        Real hInv = 1/h;
        for (int j=0; j<nd; ++j) {

          if (Niter==0) {
            val_on_first_pass[j] = fptmp[j] - f0tmp[j];
          }
          else {

            Real this_pass = fptmp[j] - f0tmp[j];
            if (!this_one_good[j] && this_pass > 10*val_on_first_pass[j]) {
              this_one_good[j] = true;
              num_reqd[j] = Niter;
              fjac[i*m+n+j] = this_pass * hInv; // Column major              
              //std::cout << "J vals: " << i << " " << j << " " << fptmp[j] << " " << num_reqd[j] << std::endl;
            }
          }
        }

        pvals[i] = x[i];

        done = true;
        for (int i=1; i<nd; ++i) { // HACK
          done &= this_one_good[i];
        }
        more_work = !done || Niter==NiterMAX-1;

      }
      
      // Verify we found good stuff for everyone
      if (!done) {
        for (int i=0; i<nd; ++i) {
          std::cout << "i " << i << " good: " << this_one_good[i] << std::endl;
        }
        BoxLib::Abort("No good twiddle found for someone");
      }

    }
#elif 0
    std::vector<Real> f0tmp(em.NumExptData());
    int ret0 = eval_nlls_data(p,pvals,&(f0tmp[0]));
    BL_ASSERT(ret0 == GOOD_EVAL_FLAG);

    for (int i=0; i<n; ++i) {
        
      Real typ = std::max(s->parameter_manager.TypicalValue(i), std::abs(pvals[i]));
      Real h = typ * s->param_eps;
      pvals[i] += h;
        
      int ret = eval_nlls_data(p,pvals,&(fptmp[0]));
      BL_ASSERT(ret == GOOD_EVAL_FLAG);
        
      Real hInv = 1/h;
      for (int j=0; j<nd; ++j) {
        fjac[i*m+n+j] = (fptmp[j] - f0tmp[j]) * hInv; // Column major
        //std::cout << "J vals: " << i << " " << j << " " << fptmp[j] << std::endl;
      }

      pvals[i] = x[i];

    }
#elif 1
    for (int i=0; i<n; ++i) {
        
      Real typ = std::max(s->parameter_manager.TypicalValue(i), std::abs(pvals[i]));
      Real h = typ * s->param_eps;

      pvals[i] = x[i] + h;
      int ret = eval_nlls_data(p,pvals,&(fptmp[0]));
      BL_ASSERT(ret == GOOD_EVAL_FLAG);
        
      pvals[i] = x[i] - h;
      ret = eval_nlls_data(p,pvals,&(fmtmp[0]));
      BL_ASSERT(ret == GOOD_EVAL_FLAG);
        
      Real hInv = 1/h;
      for (int j=0; j<nd; ++j) {
        fjac[i*m+n+j] = (fptmp[j] - fmtmp[j]) * hInv * 0.5; // Column major
      }

      pvals[i] = x[i];

    }
#endif

  }
  else {
    BoxLib::Abort("Bad iflag value");
  }
  return 0;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


MyMat
NLLSMinimizer::JTJ(void *p, const std::vector<Real>& X)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int n = s->parameter_manager.NumParams();
  int m = n + s->expt_manager.NumExptData();
  std::vector<Real> fvec(m);
  std::vector<Real> fjac(m*n);
  int status = NLLSFCN(p,m,n,&(X[0]),&(fvec[0]),&(fjac[0]),m,2);
  MyMat H(n);

  std::vector<Real> JTJ(n*n);
  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      JTJ[j*n+i] = 0;
    }
  }
  // Fill up J^t J
  for (int i=0; i<n; ++i) {
    for (int j=0; j<n; ++j) {
      for (int ii=0; ii<m; ++ii) {
        JTJ[j*n+i] += fjac[i*m+ii] * fjac[j*m+ii];
      }
    }
  }
  for( int ii=0; ii<n; ii++ ){
    H[ii].resize(n);
    for (int j=0; j<n; ++j) {
      H[ii][j] = 2*JTJ[j + ii*n];
    }
  }
  return H;
}

// /////////////////////////////////////////////////////////
// Nonlinear least squares function that does not evaluate
// the Jacobian (Minpack will do that for us)
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
static
int NLLSFCN_NOJ(void *p, int m, int n, const real *x, real *fvec, int iflag)
{
  int eflag = eval_nlls_funcs(p,m,n,x,fvec);
  if (eflag != GOOD_EVAL_FLAG) {
    if (eflag == BAD_DATA_FLAG) {
      MINPACKstruct *s = (MINPACKstruct*)(p);
      ParameterManager& pm = s->parameter_manager;
      const std::vector<Real>& upper_bound = pm.UpperBound();
      const std::vector<Real>& lower_bound = pm.LowerBound();
      std::cout << "Bad parameters" << std::endl;
      for (int i=0; i<n; ++i) {
        if (x[i] < lower_bound[i] || x[i] > upper_bound[i]) {
          std::cout << "    parameter " << i << " oob: " << x[i] << " " << lower_bound[i] << " " << upper_bound[i] << std::endl;
        }
      }
    }
    BoxLib::Abort("HANDLE BAD EVAL FLAG");
  }

  if (iflag == 0) {
    Real sum = 0;
    for (int i=0; i<m; ++i) {
      sum += fvec[i] * fvec[i];
    }
    std::cout << "X: { ";
    for (int i=0; i<n; ++i) {
      std::cout << x[i] << " ";
    }
    std::cout << "} FUNC: " << sum << std::endl;
  }
  return 0;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////



// /////////////////////////////////////////////////////////
// Our call to minpack
// This is the old call that does not use least squares
// structure. 
// Tolerances may not be set correctly.
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
GeneralMinimizer::minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln)
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
      DIAG[i] = std::abs(1/s->parameter_manager[i].DefaultValue());
    }
  }
  Real EPSFCN=1e-6;
  std::vector<Real> FJAC(num_vals*num_vals);

  Real XTOL=1.e-8;
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

};


// /////////////////////////////////////////////////////////
// Our call to Minpack for nonlinear least squares
// All tolerances were picked according to the user guide
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
NLLSMinimizer::minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln)
{
  MINPACKstruct *s = (MINPACKstruct*)(p);
  int n = s->parameter_manager.NumParams();
  int m = n + s->expt_manager.NumExptData();
  std::vector<Real> fvec(m);
  soln = guess;

#if 1
  std::vector<Real> diag(n);
  int mode = 2;
  if (mode==2) {
    for (int i=0; i<n; ++i) {
      diag[i] = std::abs(1/s->parameter_manager[i].DefaultValue());
    }
  }

  int nprint = 1;
  int maxfev = 1000;
  Real factor=100;
  int ldfjac = m;
  Real ftol = sqrt(__cminpack_func__(dpmpar)(1));
  Real xtol = sqrt(__cminpack_func__(dpmpar)(1));
  Real gtol = 0;
  std::vector<int> ipvt(n);
  std::vector<Real> qtf(n);
  std::vector<Real> wa1(n);
  std::vector<Real> wa2(n);
  std::vector<Real> wa3(n);
  std::vector<Real> wa4(m);
  std::vector<Real> fjac(m*n);
  int nfev, njev;
  /*
    the purpose of lmder is to minimize the sum of the squares of
    m nonlinear functions in n variables by a modification of
    the levenberg-marquardt algorithm. the user must provide a
    subroutine which calculates the functions and the jacobian. */

  std::cout << "Minpack uses the function lmder "<< std::endl;
  int info = lmder(NLLSFCN,p,m,n,&(soln[0]),&(fvec[0]),&(fjac[0]),ldfjac,
                   ftol,xtol,gtol, maxfev, &(diag[0]),
                   mode,factor,nprint,&nfev,&njev,&(ipvt[0]),&(qtf[0]), 
                   &(wa1[0]),&(wa2[0]),&(wa3[0]),&(wa4[0]));

  MINPACKstruct::LAPACKstruct& lapack = s->lapack_struct;
  std::vector<Real>& a = lapack.a;
  for (int r=0; r<n; ++r) {
    for (int c=0; c<n; ++c) {
      a[r*n+c] = fjac[r*n+c];
    }
  }

  // Forget about COVAR
  // Or maybe not... Marc may re-activate to figure
  // out consistency of how we index matrices
  /*
  __cminpack_func__(covar)(n,&(a[0]),n,&(ipvt[0]),xtol,&(wa1[0]));

  // Get vector of eigenvalues of inverse (J^T . J) at numerical minimum
  std::cout << "Display: (J^T J)^-1 and its eigenvalues at numerical minimum:"<< std::endl;
  lapack_int info_la = lapack.DSYEV_wrap();
  BL_ASSERT(info_la == 0);

  const std::vector<Real>& singular_values = lapack.s;
  std::cout << "Eigenvalues of J^T . J = { ";
  for (int j=0; j<n; ++j) {
    std::cout << 1/singular_values[j] << " ";
  }
  std::cout << "}\n";
  */


  std::string msg;
  switch (info)
  {
  case 0:  msg = "improper input parameters."; break;
  case 1:  msg = "actual & predicted rel reductions in sum of squares <= ftol."; break;
  case 2:  msg = "relative error between consecutive iterates <= xtol."; break;
  case 3:  msg = "conditions for info = 1 and info = 2 both hold."; break;
  case 4:  msg = "cos angle between fvec and any col in J in abs is <= gtol."; break;
  case 5:  msg = "nfev >= maxfev"; break;
  case 6:  msg = "ftol is too small. no further reduction possible"; break;
  case 7:  msg = "xtol is too small. no further improvement in x is possible"; break;
  case 8:  msg = "gtol is too small. fvec orthogonal to cols of J to macheps"; break;
  default: msg = "unknown error.";
  }

  for (int i=0; i<soln.size(); ++i) {
    s->parameter_manager[i] = soln[i];      
  }

  if (info ==0 || info>4) {
    BoxLib::Abort(msg.c_str());
  }
  else {
    std::cout << "minpack terminated: " << msg << std::endl;
  }
#else


  // Minpack-internally generated derivatives

#if 0
  /* 
     the purpose of lmdif1 is to minimize the sum of the squares of
     m nonlinear functions in n variables by a modification of the
     levenberg-marquardt algorithm. this is done by using the more
     general least-squares solver lmdif. the user must provide a
     subroutine which calculates the functions. the jacobian is
     then calculated by a forward-difference approximation. */

  std::vector<int> iwa(n);
  int lwa = m*n+5*n+m;
  std::vector<Real> wa(lwa);
  Real tol = sqrt(__cminpack_func__(dpmpar)(1));
  std::cout << "****** USING lmdif1 "<< std::endl;
  int info = lmdif1(NLLSFCN_NOJ,p,m,n,&(soln[0]),&(fvec[0]),
                    tol,&(iwa[0]),&(wa[0]),lwa);
  std::string msg;
  switch (info)
  {
  case 0:  msg = "improper input parameters."; break;
  case 1:  msg = "relative error in the sum of squares is <= tol."; break;
  case 2:  msg = "relative error between x and the solution is <= tol."; break;
  case 3:  msg = "conditions for info = 1 and info = 2 both hold."; break;
  case 4:  msg = "fvec is orthogonal to J cols to machine precision."; break;
  case 5:  msg = "nfev >= 200*(n+1)"; break;
  case 6:  msg = "tol is too small. no further reduction possible"; break;
  case 7:  msg = "tol is too small. no further improvement in x is possible"; break;
  default: msg = "unknown error.";
  }

  if (info ==0 || info>4) {
    BoxLib::Abort(msg.c_str());
  }

#else
  /*
    the purpose of lmdif is to minimize the sum of the squares of
    m nonlinear functions in n variables by a modification of
    the levenberg-marquardt algorithm. the user must provide a
    subroutine which calculates the functions. the jacobian is
    then calculated by a forward-difference approximation. */
  
  int nfev, maxfev = 200*(n+1);
  Real factor=100;
  int ldfjac = m;
  Real ftol = sqrt(__cminpack_func__(dpmpar)(1));
  Real xtol = sqrt(__cminpack_func__(dpmpar)(1));
  Real gtol = 0;
  Real epsfcn = sqrt(__cminpack_func__(dpmpar)(1));
  ParmParse pp;
  pp.query("epsfcn",epsfcn);

  std::cout << "ftol: " << ftol << ", epsfcn: " << epsfcn << std::endl;

  /* 
     Test 1: minpack converged if EuclideanNorm(F) <= (1+ftol)*EuclideanNorm(F), F=F(xsol)

     If ftol = 10**(-K), then the final residual norm EuclideanNorm(F) has K sigfigs, 
     and info is set = 1.  Danger: smaller components of D*X may have large relerrs,
     but if MODE=1, then the accuracy of the components of X is usually related to 
     their sensitivity.  Recommend: ftol = sqrt(macheps)


     Test 2: minpack converged if EuclideanNorm(D*(x-xsol)) <= xtol*EuclideanNorm(D*xsol)

     If xtol = 10**(-K), then the larger components of D*X have K sigfigs and
     info is set = 2.  Danger: smaller components of D*X may have large relerrs,
     but if MODE=1, then the accuracy of the components of X is usually related to 
     their sensitivity.  Recommend: xtol = sqrt(macheps)
   */
  std::vector<Real> diag(n);
  int mode = 2;
  if (mode==2) {
    for (int i=0; i<n; ++i) {
      diag[i] = std::abs(1/s->parameter_manager[i].DefaultValue());
    }
  }
  int nprint = 1;
  std::vector<Real> fjac(m*n);
  std::vector<int> ipvt(n);
  std::vector<Real> qtf(n);
  std::vector<Real> wa1(n), wa2(n), wa3(n), wa4(m);

  std::cout << "****** USING lmdif "<< std::endl;
  int info = lmdif(NLLSFCN_NOJ,p,m,n,&(soln[0]),&(fvec[0]),ftol,xtol,gtol,maxfev,epsfcn,
                   &(diag[0]),mode,factor,nprint,&nfev,&(fjac[0]),
                   ldfjac,&(ipvt[0]),&(qtf[0]),&(wa1[0]),&(wa2[0]),&(wa3[0]),&(wa4[0]));
  
  MINPACKstruct::LAPACKstruct& lapack = s->lapack_struct;
  std::vector<Real>& a = lapack.a;
  for (int r=0; r<n; ++r) {
    for (int c=0; c<n; ++c) {
      a[r*n+c] = fjac[r*n+c];
    }
  }

  __cminpack_func__(covar)(n,&(a[0]),n,&(ipvt[0]),xtol,&(wa1[0]));

  // Get vector of eigenvalues of inverse(J^T . J)
  lapack_int info_la = lapack.DSYEV_wrap();
  BL_ASSERT(info_la == 0);

  const std::vector<Real>& singular_values = lapack.s;
  std::cout << "Eigenvalues of J^T . J = { ";
  for (int j=0; j<n; ++j) {
    std::cout << 1/singular_values[j] << " ";
  }
  std::cout << "}\n";

  std::string msg;
  switch (info)
  {
  case 0:  msg = "improper input parameters."; break;
  case 1:  msg = "both actual and predicted relative reductions in sum of squares <= ftol"; break;
  case 2:  msg = "relative error between two consecutive iterates <= xtol."; break;
  case 3:  msg = "conditions for info = 1 and info = 2 both hold."; break;
  case 4:  msg = "the cosine of the angle between fvec and any column of the jacobian <= gtol in absolute value."; break;
  case 5:  msg = "number of calls to fcn >= maxfev"; break;
  case 6:  msg = "ftol is too small. no further reduction in the sum of squares is possible."; break;
  case 7:  msg = "xtol is too small. no further improvement in the approximate solution x is possible."; break;
  case 8:  msg = "gtol is too small. fvec is orthogonal to the columns of the jacobian to machine precision."; break;
  default: msg = "unknown error.";
  }

  if (info ==0 || info>4) {
    BoxLib::Abort(msg.c_str());
  }
#endif


  std::cout << "minpack terminated: " << msg << std::endl;
#endif
  
  /*
  Real Ffinal = NegativeLogLikelihood(soln);
  std::cout << "Ffinal: " << Ffinal << std::endl;

  int IFLAGP = 0;
  FCN(p,n,&(soln[0]),&(fvec[0]),IFLAGP);
  std::cout << "X, FVEC: " << std::endl;
  for(int ii=0; ii<n; ii++){
    std::cout << soln[ii] << " " << fvec[ii] << std::endl;
  }
  */
};
