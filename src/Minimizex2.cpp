#include <iostream>
#include <iomanip>

#include <Observation.H>
#include "cminpack.h"



// This function is in the f90 file
extern "C" {
  void test_observation(void(*)(int, const Real*,Real*),int, const Real*, Real*);
}


// Matti *(CAREFUL)*
// ******************************************************
Real funcF(double *X, int NP){
	Real F = 0;
	int ii;
	for(ii=0;ii<NP;ii++){
		F+=0.5*X[ii]*X[ii];
	}
	return F;
}


// Compute the derivative of the function funcF with respect to 
// the Kth variable (forward finite differences)
double der_ffd(const double *X,
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
                double typ = 1;
  		xdX1[K] = xdX[K]+typ*h;
 		double fx1 = funcF(xdX1,N);

 		double fx2 = funcF(xdX,N);
 		double dx  = xdX1[K]-xdX[K];
  		double fp  = (fx1-fx2)/dx;

                delete xdX;
                delete xdX1;
  		return fp;
}

// Compute the derivative of the function funcF with respect to 
// the Kth variable (centered finite differences)
double der_cfd(	const double *X,
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
                double typ = 1;
   		xdX1[K] = xdX[K]+h*typ;
    		double fx1 = funcF(xdX1,N);
    		xdX2[K] = xdX[K]-h*typ;
    		double fx2 = funcF(xdX2,N);
    		double dx = xdX1[K]-xdX2[K];
    		double fp = (fx1-fx2)/dx;    	
                delete xdX;
                delete xdX1;
                delete xdX2;
		return fp;
}	

// Gradient with finite differences
void grad(const double *X,
	  double *gradF, 
	  int N)
{
	int ii;
	for(ii=0;ii<N;ii++){  
          gradF[ii] = der_ffd(X,N,ii); 
          //gradF[ii] = der_cfd(X,N,ii); 
  	} 
}

// This is what we give to MINPACK
int FCN(void   *p,    
         int	NP,
	 const double *X,
	 double *FVEC,
	 int 	IFLAGP)
{
 	grad(X,FVEC,NP);	
        return 0;
}


// END Matti *(CAREFUL)*
// ******************************************************



int
main ()
{
  std::cout<<std::setprecision(20);

  // Matti *(CAREFUL)*
  // ******************************************************
  int ii=0;
  int N=8;
  double *X = new double[N];
  double *gradF = new double[N];
  for(ii=0;ii<N;ii++){X[ii]=4;}

  grad(X,gradF,N);
  for(ii=0;ii<N;ii++){
   std::cout << "Gradient " << gradF[ii] << std::endl;
  }

 
  // Call minpack
  int INFO,LWA=180;                                                                                  
  double TOL=1e-14,FNORM;                                                                                  
  double *FVEC = new double[N];
  double *WA   = new double[180];                                                

  //hybrd1_(FCN,&N,X,FVEC,&TOL,&INFO,WA,&LWA); 

  INFO = hybrd1(FCN,0,N,X,FVEC,TOL,WA,LWA); 

  std::cout << "INFO: " << INFO << std::endl;
  for (int i=0; i<N; ++i) { 
    std::cout << "i,f: " << i << ", " << FVEC[i] << std::endl;
  }

  return 1; 
  // ENDMatti *(CAREFUL)*
  // ******************************************************
}

