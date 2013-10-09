#include <iostream>

#include <Observation.H>
#include "cminpack.h"

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

struct ExperimentData
{
	double *data; 		// the data
        int     num_exps;	// how many data
        double *likelihood_std; // standard deviations of likelihood
        double *prior_mean;	// mean of prior
        double *prior_std; 	// standard deviations of prior
        int 	num_vals;	// number of parameters
	
	// Constructor and destructor
	ExperimentData(	double *data_, 		// the data
        		int     num_exps_,	// how many data
        		double *likelihood_std_, // standard deviations of likelihood
       			double *prior_mean_,	// mean of prior
        		double *prior_std_,	// standard deviations of prior
        		int 	num_vals_)	// number of parameters
		: 	data(data_),
			num_exps(num_exps_),
			likelihood_std(likelihood_std_),
			prior_mean(prior_mean_),
			prior_std(prior_std_),
			num_vals(num_vals_){};
	
	// Function that prints configuration to the screen
	void Info()
	{
 		int ii;
		puts(" ");
		std::cout << "Number of parameters: " << num_vals << std::endl;
		std::cout << "Number of data points: "<< num_exps << std::endl;

		puts(" ");
		std::cout << "Data: " << std::endl;
		for(ii=0;ii<num_exps;ii++){
			 std::cout << "Mean (+/- standard deviation): " << data[ii] << " +/- " << likelihood_std[ii] << std::endl;
		};

		puts(" ");
		std::cout << "Prior: " << std::endl;
		for(ii=0;ii<num_vals;ii++){
			 std::cout << "Mean (+/- standard deviation): " << prior_mean[ii] << " +/- " << prior_std[ii] << std::endl;
		};
	};

	// Function that evaluates the prior
	double prior(double *pvals)
	{
		int ii;
		double p = 0;
		for(ii=0;ii<num_vals;ii++){
			p+=(prior_mean[ii]-pvals[ii])*(prior_mean[ii]-pvals[ii])/2/prior_std[ii]/prior_std[ii];
		}	
		return p;
	}

	// Function that evaluates the likelihood
	double likelihood(double *pvals)
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

	Real funcF(double *pvals)
	{
		Real F = prior(pvals)+likelihood(pvals);
		return F;
	}


	// Compute the derivative of the function funcF with respect to 
	// the Kth variable (centered finite differences)
	double der_cfd(const double *X,
				int K)
	{
  		int ii;
  		double h = 1.5e-8;
  		double *xdX  = new double[num_vals];
  		double *xdX1 = new double[num_vals];
  		double *xdX2 = new double[num_vals];

  		for(ii=0;ii<num_vals;ii++){
     			xdX[ii]  = X[ii];
     			xdX1[ii] = X[ii];
     			xdX2[ii] = X[ii];
  		}
                
		double typ = std::abs(xdX[K]);
		if(std::abs(typ)<1){typ= 1;};

   		xdX1[K] = xdX[K]+h*typ;
    		double fx1 = funcF(xdX1);
    		xdX2[K] = xdX[K]-h*typ;
    		double fx2 = funcF(xdX2);
    		double dx = xdX1[K]-xdX2[K];
    		double fp = (fx1-fx2)/dx;    	
                delete xdX;
                delete xdX1;
                delete xdX2;
		return fp;
	}	

	// Compute the derivative of the function funcF with respect to 
	// the Kth variable (forward finite differences)
	double der_ffd(const double *X,
			        int K)
	{
		int ii;
  		double h = 1.5e-8;
  		double *xdX  = new double[num_vals];
 	 	double *xdX1 = new double[num_vals];

  		for(ii=0;ii<num_vals;ii++){
    	 		xdX[ii] = X[ii];
     			xdX1[ii]= X[ii];
  		}
                double typ = std::abs(xdX[K]);
		if(std::abs(typ)<1){typ= 1;};
		xdX1[K] = xdX[K]+typ*h;
 		double fx1 = funcF(xdX1);

 		double fx2 = funcF(xdX);
 		double dx  = xdX1[K]-xdX[K];
  		double fp  = (fx1-fx2)/dx;

                delete xdX;
                delete xdX1;
  		return fp;
	}

	// Gradient with finite differences
	void grad(const double *X,
	  		double *gradF)
	{
		int ii;
		for(ii=0;ii<num_vals;ii++){  
         		gradF[ii] = der_ffd(X,ii); 
          		//gradF[ii] = der_cfd(X,ii); 
  		} 
	}

	
	// This is what we give to MINPACK
	int FCN(void   *p,    
         	int	NP,
	 	const double *X,
	 	double *FVEC,
	 	int 	IFLAGP)
	{
 		grad(X,FVEC);	
		
		int ii;
		puts(" ");
		std::cout <<"Gradient via FNC: " << std::endl;

		for(ii=0;ii<num_vals;ii++){
  			std::cout << FVEC[ii] << std::endl;
  		};

        	return 0;
	}

	/*
	// Call minpack
	void minimize(double *InitialGuess)
	{
		int INFO,LWA=180;                                                                                  
 		double TOL=1e-14;                                                                                  
  		double *FVEC = new double[num_vals];
  		double *WA   = new double[180];                                                

		InitialGuess = prior_mean;

		INFO = hybrd1(FCN,0,num_vals,InitialGuess,FVEC,TOL,WA,LWA); 

  		std::cout << "INFO: " << INFO << std::endl;
  		for (int i=0; i<num_vals; ++i) { 
    		std::cout << "i,f: " << i << ", " << FVEC[i] << std::endl;
 		}
	};
	*/


	/*
	// Generate a sample of the prior (i.e. a set of parameters)
	void generate_prior_sample(double *sample)
	{
		int ii;
		for(ii=0;ii<num_vals;ii++){
			sample[ii]=prior_mean[ii]+prior_std[ii]*randn();
		}
	}



	void PriorImportanceSampling(int NOS) // input is number of samples
	{
 		double *sample = new double[num_vals];
		double *weights = new double[num_samples];
  		int    ii;
  		for(ii=0;ii<NOS;ii++){
			generate_prior_sample(sample);
			weights[ii] = likelihood(sample);
		}
	};
	*/
};

//ExperimentData expData;

#if 0
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
  //
  int num_exps = 1; 			 	     	// how many data points/different experiments do we have?
  double *true_params = pdata.dataPtr();		// True parameter set	
  int ii;
 
  // Output true parameters (we want to find these via sampling) 	  
  std::cout << "True parameters: " << std::endl;
  for(ii=0;ii<num_vals;ii++){
	  std::cout << true_params[ii] << std::endl;
  };
  
  // Define the likelihood
  double *likelihood_std = new double[num_exps];
  for(ii=0;ii<num_exps;ii++){ 
	  likelihood_std[ii] = 14.; // standard deviations
  }	
 
  // Define the prior, by giving it a mean and variance for all the active parameters
  // Std is 10% of the true value of the parameter
  // Mean is 0.5*std away from true value
  double *prior_mean = new double[num_vals];
  double *prior_std  = new double[num_vals];
  for(ii=0;ii<num_vals;ii++){
	  prior_std[ii] = true_params[ii]*0.1;
	  if(prior_std[ii] == 0){prior_std[ii] =1e-2;}
	  prior_mean[ii] = true_params[ii]*(1+0.5*prior_std[ii]);
	  if(prior_mean[ii] == 0){prior_mean[ii] =1e-2;}

  }

  // generate a synthetic set of data: run observation with true parameters and perturb what we get  
  double *data = new double[num_exps];
  generate_data(data,num_exps,likelihood_std,num_vals,true_params);

  // Creat an "ExperimentData" instant
  ExperimentData ExpData(data,num_exps,likelihood_std,prior_mean, prior_std,num_vals);	
  ExpData.Info();	
  
  // Test functions in class
  puts(" ");
  std::cout << "Prior evaluated at prior mean: " <<  ExpData.prior(prior_mean) << std::endl;
  std::cout << "Prior evaluated at true parameters: " <<  ExpData.prior(true_params) << std::endl;
  
  puts(" ");
  std::cout << "Likelihood evaluated at prior mean: " <<  ExpData.likelihood(prior_mean) << std::endl;
  std::cout << "Likelihood evaluated at true parameters:  " <<  ExpData.likelihood(true_params) << std::endl; 
  
  puts(" ");
  std::cout << "F evaluated at prior mean:   " <<  ExpData.funcF(prior_mean) << std::endl; 
  std::cout << "F evaluated at true parameters:  " <<  ExpData.funcF(true_params) << std::endl;
  
  double *gradF = new double[num_vals];
  ExpData.grad(prior_mean,gradF);
  puts(" ");
  std::cout << "Gradient evaluated at true parameters:  "<< std::endl;
  for(ii=0;ii<num_vals;ii++){
  	std::cout << gradF[ii] << std::endl;
  };

  void *p;
  int FLAG;
  int tmp = ExpData.FCN(p,num_vals,prior_mean,gradF,FLAG); 
  std::cout << "FCN evaluated at true parameters:  "<< tmp << std::endl;





/*
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
*/
  
  
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

