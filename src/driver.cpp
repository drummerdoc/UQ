#include <iostream>

#include <Observation.H>
//#include "minpack.h"

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

// This function is in the f90 file
extern "C" {
  void test_observation(void(*)(int, const Real*,Real*),int, const Real*, Real*);
}


// Matti *(CAREFUL)*
// ******************************************************
// Generate uniformly distributed random number 
Real drand(){
  return (rand()+1.0)/(RAND_MAX+1.0);
}

// Generate standard normal random number 
Real randn(){
  double pi; 
  pi =  3.14159265358979323846;
  return sqrt(-2*log(drand())) * cos(2*pi*drand());
}

// Make experiment(s) and perturb it(them) to simulate measurement error
// These "data" are given to minpack
void
generate_data(double *data, // the "output" is data, array of length num_exps
	      int num_exps, // How many "experiments", or "data points" do we have
	      double *likelihood_std,
	      int num_vals, const Real* pvals) // input parameters of observation function(s)
{
  		Real y;
 		observation_function(num_vals,pvals,&y);// This may become a loop over various obseration_functions
  		int ii;
  		for(ii=0;ii<num_exps;ii++){
	 		data[ii] = y+likelihood_std[ii]*randn();
  		}
}


// Generate a sample of the prior (i.e. a set of parameters)
void generate_prior_sample(
		double 	*sample, 		// the sample
		int 	 num_vals, 		// number of parameters
		double  *prior_mean,		// the mean
		double  *prior_std)	// the standard deviations
{
	int ii;
	for(ii=0;ii<num_vals;ii++){
		sample[ii]=prior_mean[ii]+prior_std[ii]*randn();
	}
}

// Function that evaluates the prior
double prior(	double *prior_mean,	// mean of prior
		double *prior_std, 	// standard deviations of prior
		double *pvals,		// the parameters
		int 	num_vals)	// number of parameters
{
	int ii;
	double p = 0;
	for(ii=0;ii<num_vals;ii++){
		p+=(prior_mean[ii]-pvals[ii])*(prior_mean[ii]-pvals[ii])/2/prior_std[ii]/prior_std[ii];
	}	
	return p;
}

// Function that evaluates the likelihood
double likelihood(
	double *data, 		// the data
    	int     num_exps,	// how many data
	double *likelihood_std, // standard deviations of likelihood
	double *pvals,		// the parameters
	int 	num_vals)	// number of parameters
{
	Real ret;
	observation_function(num_vals,pvals,&ret);
	double l = 0;
	int ii;
	for(ii=0;ii<num_exps;ii++){
		l+=(data[ii]-ret)*(data[ii]-ret)/2/likelihood_std[ii]/likelihood_std[ii];
	}
	return l;
}

#if 0
struct ExperimentData
{
  ExperimentDatafuncF()
  void initialize(	double *data, 		// the data
                        int     num_exps,	// how many data
                        double *likelihood_std, // standard deviations of likelihood
                        double *prior_mean,	// mean of prior
                        double *prior_std, 	// standard deviations of prior
                        int 	num_vals)	// number of parameters
  double *data; // the data
  int     num_exps; // how many data
  double *likelihood_std; // standard deviations of likelihood
  double *prior_mean; // mean of prior
  double *prior_std; // standard deviations of prior
  int 	num_vals; // number of parameters
};

ExperimentData expData;


  extern "C" {
    void
    funcF()
    {
      double* the_data = expData.data;
      
    }
  }


main() {

  expData.initialize(...)

  minpack
...


}

#endif

// The function F we will minimize with Minpack 
// Note: we will give Minpack the gradient of this function
Real 
funcF(	double *data, 		// the data
    	int     num_exps,	// how many data
	double *likelihood_std, // standard deviations of likelihood
	double *prior_mean,	// mean of prior
	double *prior_std, 	// standard deviations of prior
	double *pvals,		// the parameters
	int 	num_vals)	// number of parameters
{
	Real F = 0;
	Real ret;
	int ii;
	// prior
	for(ii=0;ii<num_vals;ii++){
		F+= prior(prior_mean,prior_std,pvals,num_vals);
	}
	// likelihood
	for(ii=0;ii<num_exps;ii++){
		observation_function(num_vals,pvals,&ret);
		F+=likelihood(data,num_exps,likelihood_std,pvals,num_vals);
	}
	return F;
}

/*
Real funcF(double *X){
	Real F = 0;
	int ii;
	for(ii=0;ii<8;ii++){
		F+=0.5*X[ii]*X[ii];
	}
	return F;
}


// Compute the derivative of the function funcF with respect to 
// the Kth variable (forward finite differences)
double der_ffd(double *X,
	       int N, 
	       int K)
{
		int ii;
  		double h=1.5e-8;
  		double *xdX  = new double[N];
 	 	double *xdX1 = new double[N];

  		for(ii=0;ii<N;ii++){
    	 		xdX[ii] = X[ii];
     			xdX1[ii]= X[ii];
  		}
  		xdX1[K] = xdX[K]+h*xdX[K];
 		double fx1 = funcF(xdX1);
 		double fx2 = funcF(xdX);
 		double dx  = xdX1[K]-xdX[K]; 
  		double fp  = (fx1-fx2)/dx;
  		return fp;
}

// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
double der_cfd(	double *X,
		int N, 
		int K)
{
  		int ii;
  		double h=1.5e-8;
  		double *xdX  = new double[N];
  		double *xdX1 = new double[N];
  		double *xdX2 = new double[N];

  		for(ii=0;ii<N;ii++){
     			xdX[ii]  = X[ii];
     			xdX1[ii] = X[ii];
     			xdX2[ii] = X[ii];
  		}
   		xdX1[K] = xdX[K]+h*xdX[K];
    		double fx1 = funcF(xdX1);
    		xdX2[K] = xdX[K]-h*xdX[K];
    		double fx2 = funcF(xdX2);
    		double dx = xdX1[K]-xdX2[K]; 
    		double fp = (fx1-fx2)/dx;
    		return fp;
}	

// Gradient with finite differences
void grad(double *X,
	  double *gradF, 
	  int N)
{
	int ii;
	for(ii=0;ii<N;ii++){  
   		//gradF[ii] = der_ffd(X,N,ii); 
		gradF[ii] = der_cfd(X,N,ii); 
  	} 
}

// This is what we give to MINPACK
void FCN(int 	*NP,
	double 	*X,
	double 	*FVEC,
	int 	*IFLAGP)
{
 	int ii;
	int n=8;	
 	double *gradF = new double[n];	
 	grad(X,gradF,n);	
 	for(ii=0;ii<n;ii++) FVEC[ii]=gradF[ii];
}
*/
// END Matti *(CAREFUL)*
// ******************************************************



int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  if (argc<2) print_usage(argc,argv);
    
  Observation ctx;

  // Populate UserContext with parameters, initialize
  // array of Reals with default values (returned by AddParameter)
  Array<Real> pdata;
  pdata.push_back(ctx.AddParameter(8,ChemDriver::FWD_BETA));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::FWD_EA));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::LOW_A));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::LOW_BETA));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::LOW_EA));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::TROE_A));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::TROE_TS));
  pdata.push_back(ctx.AddParameter(8,ChemDriver::TROE_TSSS));
  int num_vals = ctx.NumParameters();


  // Matti *(CAREFUL)*
  // ******************************************************
  // This "generates the data", i.e. it runs the observation function(s)
  // and perturbs them by a Gaussian with mean zero and known variance.
  // This variance is also used in minpack later on.
  // The reason it is here is that it needs the above stuff and the observation
  // function (but currently only one observation object can be created.
  // What it does: Takes in the true parameters/conficuration of ChemKin
  // and outputs the "data". This corresponds to a simulation of going to a lab,
  // performing an experiment, measuring something, writing it down and using the
  // result for the parameter estimation. This has nothing to do with the estimation itself,
  // but the "synthetic data" created by gen_data is used during the estimation process.
  //
  int ii;
  int num_exps = 1; 			 	     	// how many data points/different experiments do we have?
  double *data = new double[num_exps];
  double *likelihood_std = new double[num_exps];	// What is the error in the measurement processes?
  					  		// For now we think of the errors as being uncorrelated,
					 		// so that likelihood_stds is a vector of length num_exps;
  double *true_params = pdata.dataPtr();		// True parameter set						 
  Real ret;
  
  // Here I set the likelihood
  for(ii=0;ii<num_exps;ii++){ 
	  likelihood_std[ii] = 14.; // standard deviations
  }	
 
  // Here I define the prior, by giving it a mean and variance for all the active parameters
  // Std is 10% of the true value of the parameter
  // Mean is 0.5*std away from true value
  std::cout << "Prior:\n"; 
  double *prior_mean = new double[num_vals];
  double *prior_std  = new double[num_vals];
  for(ii=0;ii<num_vals;ii++){
	  prior_std[ii] = true_params[ii]*0.1;
	  if(prior_std[ii] == 0){prior_std[ii] =1e-2;}
	  prior_mean[ii] = true_params[ii]*(1+0.5*prior_std[ii]);
	  if(prior_mean[ii] == 0){prior_mean[ii] =1e-2;}

  }
  // Output true parameters and prior	  
  for(ii=0;ii<num_vals;ii++){
	  std::cout << "True: " << true_params[ii] << " Mean: " << prior_mean[ii] << "  Standard deviation: " << prior_std[ii] << std::endl;
  }

  // Observation function with true parameters and without added noise 
  observation_function(num_vals,true_params,&ret);
  for(ii=0;ii<num_exps;ii++){
	 std::cout << "Unperturbed value of obs with true parameters: " << ret << std::endl;
  }
  // generate a synthetic set of data: run observation with true parameters and perturb what we get  
  generate_data(data,num_exps,likelihood_std,num_vals,true_params);
  for(ii=0;ii<num_exps;ii++){
	  std::cout << "Data (output of observation funciton perturbed): " << data[ii] << std::endl;
  }	
  // run observation with prior mean
  observation_function(num_vals,prior_mean,&ret);
  for(ii=0;ii<num_exps;ii++){
	  std::cout << "Data with prior parameters: " << ret << std::endl;
  }

  // The function F
  Real F = funcF(data,num_exps,likelihood_std,prior_mean,prior_std,prior_mean,num_vals);  	
  std::cout << "F =  " << F << std::endl;
  
  // generate samples of prior and compare to data (simple MC scheme)
  double *sample = new double[num_vals];
  int 	  num_samples = 1000;
  double *weights = new double[num_samples];
  int     kk;
  for(ii=0;ii<100;ii++){
	for(kk=0;kk<num_vals;kk++){
		generate_prior_sample(sample,num_vals,prior_mean,prior_std);
	}
	observation_function(num_vals,sample,&ret);
	weights[ii] = likelihood(data,num_exps,likelihood_std,sample,num_vals);
	std::cout << "Outcome: " << ret << " Weight: " << weights[ii] <<std::endl;
  }

  
  
  /*
  double *X 	= new double[num_vals];
  for(ii=0;ii<num_vals;ii++){X[ii]=1;}		
  double *gradF = new double[num_vals];

  grad(X,gradF,num_vals);
  for(ii=0;ii<num_vals;ii++){
	   std::cout << "Gradient " << gradF[ii] << std::endl;
  }
  */
  // generate 100 samples of prior and look what data they give
  /*
  double *sample = new double[num_vals];
  int kk;
  for(ii=0;ii<100;ii++){
	for(kk=0;kk<num_vals;kk++){
		generate_prior_sample(sample,num_vals,prior_mean,prior_std);
	}
	observation_function(num_vals,sample,&ret);
	std::cout << "Outcome: " << ret << std::endl;
  }
  */

  // Call minpack
  /*
  int INFO,LWA=180;                                                                                  
  double TOL=1e-7,FNORM;                                                                                  
  double *FVEC = new double[8];
  double *WA   = new double[180];                                                

  hybrd1_(FCN,&num_vals,X,FVEC,&TOL,&INFO,WA,&LWA);  
  */
  // ENDMatti *(CAREFUL)*
  // ******************************************************




  // Call observation function from C++
  //Real ret;
  //observation_function(cnt,pdata.dataPtr(),&ret);
  //std::cout << "Observation (C++): " << ret << std::endl;

  // Call observation function via function ptr passed to f90 routine
  //test_observation(&observation_function,cnt,pdata.dataPtr(),&ret);
  //std::cout << "Observation (F90): " << ret << std::endl;

  BoxLib::Finalize();
}

