#include <Driver.H>
#include <Sampler.H>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

#ifdef _OPENMP
#include "omp.h"
#endif

PriorMCSampler::PriorMCSampler(const std::vector<Real>& prior_mean_,
                               const std::vector<Real>& prior_std_)
  : prior_mean(prior_mean_), prior_std(prior_std_)
{}

void
PriorMCSampler::Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  str->ResizeWork();
	  
  int num_params = str->parameter_manager.NumParams();
  int NOS = samples.size();

  std::vector<Real> sample_data(str->expt_manager.NumExptData());
  std::vector<Real> s(num_params);
  
  std::cout <<  " " << std::endl;
  std::cout <<  "Start sampling with prior " << std::endl;
  std::cout <<  "Number of samples: " << NOS << std::endl;

  const std::vector<Real>& upper_bound = str->parameter_manager.UpperBound();
  const std::vector<Real>& lower_bound = str->parameter_manager.LowerBound();

  for(int ii=0; ii<NOS; ii++){
    BL_ASSERT(samples[ii].size()==num_params);
    for(int jj=0; jj<num_params; jj++){
      samples[ii][jj] = prior_mean[jj] + prior_std[jj]*randn();
      bool sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj]);

      while (sample_oob) {
        std::cout <<  "sample is out of bounds, parameter " << jj
                  << " val,lb,ub: " << samples[ii][jj]
                  << ", " << lower_bound[jj] << ", " << upper_bound[jj] << std::endl;
        samples[ii][jj] = prior_mean[jj] + prior_std[jj]*randn();
        sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj]);
      }
    }
  }

#ifdef _OPENMP
  int tnum = omp_get_max_threads();
  BL_ASSERT(tnum>0);
  std::cout <<  " number of threads: " << tnum << std::endl;
  int logPeriod = 1000;
#else
  int tnum = 1;
#endif
  Real dthread = NOS / Real(tnum);
  std::vector<int> trange(tnum);
  for (int ithread = 0; ithread < tnum; ithread++) {
    trange[ithread] = ithread * dthread;
  }

  std::cout <<  "Generating samples";
#ifdef _OPENMP
  std::cout << " using " << tnum << " threads";
#endif
  std::cout << std::endl;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int ithread = 0; ithread < tnum; ithread++) {
    int iBegin = trange[ithread];
    int iEnd = (ithread==tnum-1 ? NOS : trange[ithread+1]);

    for(int ii=iBegin; ii<iEnd; ii++){

#ifdef _OPENMP
      if (ithread == 0) {
        if ( (ii - iBegin)%logPeriod == 0 ) {
          std::cout << " Completed "<< ii - iBegin << " samples on thread 0" << std::endl;          
        }
      }
#endif

      bool ok = str->expt_manager.GenerateTestMeasurements(samples[ii],sample_data);
      w[ii] = (ok ? str->expt_manager.ComputeLikelihood(sample_data) : -1);
    }
  }


  Real wmin = w[0];
  for(int ii=0; ii<NOS; ii++){
    wmin = std::min(w[ii],wmin);
  }
  for(int ii=0; ii<NOS; ii++){
    w[ii] = (w[ii] == -1 ? 0 : std::exp(-(w[ii] - wmin)));
    //std::cout << "w before = "<< w[ii] << std::endl;
  }

  // Normalize weights, print to terminal
  NormalizeWeights(w);
  
  // Approximate effective sample size and quality measure R	
  Real Neff = EffSampleSize(w,NOS);
  std::cout <<  " " << std::endl;
  std::cout <<  "Effective sample size = "<< Neff << std::endl;
  Real R = CompR(w,NOS);
  std::cout <<  "Quality measure R =  "<< R << std::endl;

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
  WriteSamplesWeights(samples,w,"PriorMCSampler");

  // Resampling
  std::vector<std::vector<Real> > Xrs(NOS, std::vector<Real>(num_params,-1));// resampled parameters
  Resampling(Xrs,w,samples);
  WriteResampledSamples(Xrs,"PriorMCSampler");

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
  std::cout <<  "End sampling with prior" << std::endl;
  std::cout <<  " " << std::endl;
}


LinearMapSampler::LinearMapSampler(const std::vector<Real>& mu_,
                                   const MyMat& H_,
                                   const MyMat& invsqrt_,
                                   Real phi_)
  : mu(mu_), H(H_), invsqrt(invsqrt_), phi(phi_)
{}

void
LinearMapSampler::Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  str->ResizeWork();
	  
  int num_params = str->parameter_manager.NumParams();
  int NOS = samples.size();

  std::vector<Real> sample_data(str->expt_manager.NumExptData());
  std::vector<Real> s(num_params);
  std::vector<Real> Fo(NOS);
  
  std::cout <<  " " << std::endl;
  std::cout <<  "Starting linear map sampler " << std::endl;
  std::cout <<  "Number of samples: " << NOS << std::endl;

  const std::vector<Real>& upper_bound = str->parameter_manager.UpperBound();
  const std::vector<Real>& lower_bound = str->parameter_manager.LowerBound();

  for(int ii=0; ii<NOS; ii++){
    BL_ASSERT(samples[ii].size()==num_params);

    for(int jj=0; jj<num_params; jj++){
      samples[ii][jj] = mu[jj];
      for(int kk=0; kk<num_params; kk++){
        samples[ii][jj] += invsqrt[jj][kk]*randn();
      }
      
      bool sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj]);

      while (sample_oob) {
        std::cout <<  "sample is out of bounds, parameter " << jj
                  << " val,lb,ub: " << samples[ii][jj]
                  << ", " << lower_bound[jj] << ", " << upper_bound[jj] << std::endl;

        samples[ii][jj] = mu[jj];
        for(int kk=0; kk<num_params; kk++){
          samples[ii][jj] += invsqrt[jj][kk]*randn();
        }
        sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj]);
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
      w[ii] = -Fo[ii] + F;
    }
  }
  Real wmin = w[0];
  for(int ii=0; ii<NOS; ii++){
    wmin = std::min(w[ii],wmin);
  }
  for(int ii=0; ii<NOS; ii++){
    w[ii] = (w[ii] == -1 ? 0 : std::exp(-(w[ii] - wmin)));
  }

  // Normalize weights, print to terminal
  NormalizeWeights(w);
  
  // Approximate effective sample size	
  Real Neff = EffSampleSize(w,NOS);
  std::cout <<  " " << std::endl;
  std::cout <<  "Effective sample size = "<< Neff << std::endl;
  Real R = CompR(w,NOS);
  std::cout <<  "Quality measure R = "<< R << std::endl;

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
  WriteSamplesWeights(samples, w,"LinearMapSampler");

  // Resampling
  std::vector<std::vector<Real> > Xrs(NOS, std::vector<Real>(num_params,-1));// resampled parameters
  Resampling(Xrs,w,samples);
  WriteResampledSamples(Xrs,"LinearMapSampler");

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
  std::cout <<  "End linear map sampler" << std::endl;
  std::cout <<  " " << std::endl;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


void
SymmetrizedLinearMapSampler::Sample(void* p,std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const
{
  MINPACKstruct *str = (MINPACKstruct*)(p);
  str->ResizeWork();
	  
  int num_params = str->parameter_manager.NumParams();
  int NOS = samples.size();

  std::vector<Real> sample_data(str->expt_manager.NumExptData());
  std::vector<Real> s(num_params);
  std::vector<Real> Fo(NOS);
  std::vector<std::vector<Real> > negsamples(NOS, std::vector<Real>(num_params,-1));

  
  std::cout <<  " " << std::endl;
  std::cout <<  "Starting symmetrized linear map sampler " << std::endl;
  std::cout <<  "Number of samples: " << NOS << std::endl;

  const std::vector<Real>& upper_bound = str->parameter_manager.UpperBound();
  const std::vector<Real>& lower_bound = str->parameter_manager.LowerBound();

  for(int ii=0; ii<NOS; ii++){
    BL_ASSERT(samples[ii].size()==num_params);

    for(int jj=0; jj<num_params; jj++){
      samples[ii][jj] = mu[jj];
      negsamples[ii][jj] = mu[jj];
      for(int kk=0; kk<num_params; kk++){
	real tmp = randn();
        samples[ii][jj]    += invsqrt[jj][kk]*tmp;
	negsamples[ii][jj] -= invsqrt[jj][kk]*tmp;
      }
      
      bool sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj] || negsamples[ii][jj] < lower_bound[jj] || negsamples[ii][jj] > upper_bound[jj]);

      while (sample_oob) {
        std::cout <<  "sample is out of bounds, parameter " << jj
                  << " val,lb,ub: " << samples[ii][jj]
                  << ", " << lower_bound[jj] << ", " << upper_bound[jj] << std::endl;

        samples[ii][jj] = mu[jj];
        for(int kk=0; kk<num_params; kk++){
 	  real tmp = randn();
          samples[ii][jj]    += invsqrt[jj][kk]*tmp;
	  negsamples[ii][jj] -= invsqrt[jj][kk]*tmp;
        }
        sample_oob = (samples[ii][jj] < lower_bound[jj] || samples[ii][jj] > upper_bound[jj] || negsamples[ii][jj] < lower_bound[jj] || negsamples[ii][jj] > upper_bound[jj]);
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
      Real negF = NegativeLogLikelihood(negsamples[ii]);
      w[ii] = -Fo[ii] + F - std::log( 1+std::exp(F-negF) );
      // pick -x with probability w(x)/(w(x)+w(-x))
      Real tmp = drand();
      if(  tmp < 1 / (1+std::exp(F-negF))){
         samples[ii] = negsamples[ii];	      
      }
    }
  }
  Real wmin = w[0];
  for(int ii=0; ii<NOS; ii++){
    wmin = std::min(w[ii],wmin);
  }
  for(int ii=0; ii<NOS; ii++){
    w[ii] = (w[ii] == -1 ? 0 : std::exp(-(w[ii] - wmin)));
  }

  // Normalize weights, print to terminal
  NormalizeWeights(w);
  
  // Approximate effective sample size	
  Real Neff = EffSampleSize(w,NOS);
  std::cout <<  " " << std::endl;
  std::cout <<  "Effective sample size = "<< Neff << std::endl;
  Real R = CompR(w,NOS);
  std::cout <<  "Quality measure R = "<< R << std::endl;

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
  WriteSamplesWeights(samples, w,"SymmetrizedLinearMapSampler");

  // Resampling
  std::vector<std::vector<Real> > Xrs(NOS, std::vector<Real>(num_params,-1));// resampled parameters
  Resampling(Xrs,w,samples);
  WriteResampledSamples(Xrs,"SymmetrizedLinearMapSampler");

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
  std::cout <<  "End linear map sampler" << std::endl;
  std::cout <<  " " << std::endl;
}

// /////////////////////////////////////////////////////////
// Normalize weights to that their sum is 1
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::NormalizeWeights(std::vector<Real>& w)
{
  int NOS = w.size();
  Real SumWeights = 0;
  for(int ii=0; ii<NOS; ii++){
	SumWeights = SumWeights+w[ii];
  }
  for(int ii=0; ii<NOS; ii++){
	w[ii] = w[ii]/SumWeights;
  }
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// Compute the mean of a scalar from samples
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::ScalarMean(Real& Mean, std::vector<Real> & samples)
{
  int NOS = samples.size();
  Mean = 0; // initialize  
  for(int jj=0; jj<NOS; jj++){
	  Mean = Mean+samples[jj];
  }	  
  Mean = Mean/NOS;	
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// Compute the mean of a vector from samples
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::Mean(std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples)
{
  int NOS = samples.size();
  int num_params = samples[1].size();
  for(int ii=0; ii<num_params; ii++){
	  Mean[ii] = 0; // initialize  
	  for(int jj=0; jj<NOS; jj++){
		  Mean[ii] = Mean[ii]+samples[jj][ii];
	  }
	  Mean[ii] = Mean[ii]/NOS;
  }	
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




// /////////////////////////////////////////////////////////
// Compute the weighted mean of a vector from samples
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::WeightedMean(std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples)
{
  int NOS = samples.size();
  int num_params = samples[1].size();
  for(int ii=0; ii<num_params; ii++){
	  CondMean[ii] = 0; // initialize  
	  for(int jj=0; jj<NOS; jj++){
		  CondMean[ii] = CondMean[ii]+w[jj]*samples[jj][ii];
	  }
  }	
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




// /////////////////////////////////////////////////////////
// Compute the covariance matrix from samples
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::Var(std::vector<Real>& Var,std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples)
{
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////


// /////////////////////////////////////////////////////////
// Compute covariance from weighted samples
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::WeightedVar(std::vector<Real>& CondVar,std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples)
{
  int NOS = samples.size();
  int num_params = samples[1].size();	
  for(int ii=0; ii<num_params; ii++){	  
  CondVar[ii] = 0;
  for(int jj=0; jj<NOS; jj++){
	  CondVar[ii] = CondVar[ii] + w[jj]*(samples[jj][ii]-CondMean[ii])*(samples[jj][ii]-CondMean[ii]);
  }
  }
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




// /////////////////////////////////////////////////////////
// Compute quality measure of ensemble
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
Real
Sampler::CompR(std::vector<Real>& w, int NOS)
{
   std::vector<Real> w2(NOS);
   for(int ii=0; ii<NOS; ii++){
	   w2[ii] = w[ii]*w[ii];
   }
   Real MeanW2;
   ScalarMean(MeanW2, w2);
   std::cout << "Mean w^2 = "<< MeanW2 << std::endl;
   Real MeanW;
   ScalarMean(MeanW, w);
   std::cout << "Mean w = "<< MeanW << std::endl;
   Real R = MeanW2/(MeanW*MeanW);
   return R;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////





// /////////////////////////////////////////////////////////
// Approximate effective sample size
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
Real
Sampler::EffSampleSize(std::vector<Real>& w, int NOS)
{
   // Approximate effective sample size
   Real SumSquaredWeights = 0;
   for(int ii=0; ii<NOS; ii++){
	   SumSquaredWeights = SumSquaredWeights + w[ii]*w[ii];
   }
   Real Neff = 1/SumSquaredWeights; 
   return Neff;
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////





// /////////////////////////////////////////////////////////
// Write samples and weights to file
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::WriteSamplesWeights(std::vector<std::vector<Real> >& samples, std::vector<Real>& w, const char *tag)
{
  std::stringstream SampleStr;
  SampleStr<<tag<<"Samples.dat";
  std::stringstream WeightStr;
  WeightStr<<tag<<"Weights.dat";
  int NOS = samples.size();
  int num_params = samples[1].size();
  std::ofstream of,of1;
  of.open(SampleStr.str().c_str());
  of1.open(WeightStr.str().c_str());
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////





// /////////////////////////////////////////////////////////
// Resampling
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::Resampling(std::vector<std::vector<Real> >& Xrs,std::vector<Real>& w,std::vector<std::vector<Real> >& samples)
{
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////





// /////////////////////////////////////////////////////////
// Write samples to file
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
void
Sampler::WriteResampledSamples(std::vector<std::vector<Real> >& Xrs, const char *tag)
{
  std::stringstream SampleStr;
  SampleStr<<tag<<"ResampledSamples.dat";
  int NOS = Xrs.size();
  int num_params = Xrs[1].size();
  std::ofstream of2;
  of2.open(SampleStr.str().c_str());
  of2 << std::setprecision(20);
  for(int jj=0;jj<NOS;jj++){
	  for(int ii=0;ii<num_params;ii++){
		  of2 << Xrs[jj][ii] << " ";
	  }
	  of2 << '\n';
  }
  of2.close();
}
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////




// /////////////////////////////////////////////////////////
// Quadratic approximation of F
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
Real
Sampler::F0(const std::vector<Real>& sample,
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
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////

