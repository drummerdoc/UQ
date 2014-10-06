#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>

#include <iostream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>

#include <PremixSol.H>
#include <ParallelDescriptor.H>

int
main (int   argc,
      char* argv[])
{
#ifdef BL_USE_MPI
  MPI_Init (&argc, &argv);
  Driver driver(argc,argv,MPI_COMM_WORLD);
#else
  Driver driver(argc,argv);
#endif

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  
  const std::vector<Real>& true_data = expt_manager.TrueData();
  const std::vector<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();
  const std::vector<Real>& true_data_std = expt_manager.ObservationSTD();

  int num_data = true_data.size();
  std::cout << "True and noisy data: (npts=" << num_data << ")\n"; 
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
              << "  Standard deviation: " << prior_std[ii] 
	      << "  Difference / std: "   <<  (true_params[ii]-prior_mean[ii])/prior_std[ii] 
	      << std::endl;
  }

  Real Ftrue = NegativeLogLikelihood(true_params);
  std::cout << "F at true parameters = " << Ftrue << std::endl;
  
  ParmParse pp;

  Real F = NegativeLogLikelihood(prior_mean);
  std::cout << "F at prior mean = " << F << std::endl;

  bool use_nlls_minimizer = true;
  Minimizer* minimizer = 0;
  if (use_nlls_minimizer) {
    minimizer = new NLLSMinimizer();
  }
  else {
    minimizer = new GeneralMinimizer();
  }

  std::vector<Real> guess_params(num_params);
  std::vector<Real> soln_params(num_params);
  for(int i=0; i<num_params; i++){
    guess_params[i] = prior_mean[i];
  }
  minimizer->minimize((void*)(driver.mystruct), guess_params, soln_params);
  delete minimizer;

  std::cout << "Final parameters: " << std::endl;
  for(int ii=0; ii<num_params; ii++){
    std::cout << parameter_manager[ii] << std::endl;
  }
  // Output value of objective function at minimum
  Real phi = NegativeLogLikelihood(soln_params);
  std::cout << "F at numerical minimum = " << phi << std::endl;


  // ////////////////////////////////////////////////////////////////////
  // Do sampling
  // ////////////////////////////////////////////////////////////////////
  int NOS = 10000; pp.query("NOS",NOS);
  std::vector<Real> w(NOS);
  std::vector<std::vector<Real> > samples(NOS, std::vector<Real>(num_params,-1));

  std::string which_sampler = "prior_mc"; pp.query("which_sampler",which_sampler);
  bool fd_Hessian = true; pp.query("fd_Hessian",fd_Hessian);

  Sampler *sampler = 0;
  if (which_sampler == "prior_mc") {
    sampler = new PriorMCSampler(prior_mean, prior_std);
  }
  else {
    MyMat H, InvSqrtH;
    Real Hessian_cutoff = 1e-16; pp.query("Hessian_cutoff",Hessian_cutoff);
    if (fd_Hessian) {
      // ////////////////////////////////////////////////////////////////////
      // Compute Finite Difference Hessian
      // ////////////////////////////////////////////////////////////////////
      H = Minimizer::FD_Hessian((void*)driver.mystruct, soln_params);
      for(int i=0;i<num_params;i++){
        for(int j=0;j<i;j++){
          H[i][j]=H[j][i];
        }
      }
    }
    else {
      // ////////////////////////////////////////////////////////////////////
      // Compute J^t J
      // ////////////////////////////////////////////////////////////////////
      int n = num_params;
      int m = num_data + n;
      std::vector<Real> FVEC(m);
      std::vector<Real> FJAC(m*n);
      // Call Minpack to get Jacobian
      BoxLib::Abort("Jacobian function not yet exposed");
      if (!use_nlls_minimizer) {
        BoxLib::Abort("No wrapper to get Jacobian from function supplied to non-nlls minpack...");
      }
      //int ldfjac = m;
      //NLLSFCN((void*)(driver.mystruct),m,n,&(soln_params[0]),&(FVEC[0]),&(FJAC[0]),ldfjac,2);
 
      // Initialize J^t J
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
            JTJ[j*n+i] += FJAC[i*m+ii] * FJAC[j*m+ii];
          }
        }
      }
      // Get Hessian from J^t J and convert to a MyMat 
      for( int ii=0; ii<n; ii++ ){
        H[ii].resize(n);
        for (int j=0; j<n; ++j) {
          H[ii][j] = 2*JTJ[j + ii*n];
        }
      }
    }

    InvSqrtH = Minimizer::InvSqrt((void*)driver.mystruct, H, Hessian_cutoff);

    if (which_sampler == "linear_map") {
      sampler = new LinearMapSampler(soln_params,H,InvSqrtH,phi);
    }
    else if (which_sampler == "symmetrized_linear_map") {
      sampler = new SymmetrizedLinearMapSampler(soln_params,H,InvSqrtH,phi);
    }
    else if (which_sampler == "none") {
    }
    else {
      BoxLib::Abort("Invalid value for which_sampler");
    }
  }
  if (sampler) {
    sampler->Sample((void*)(driver.mystruct), samples, w);
  }
  delete sampler;

  BoxLib::Finalize();

#ifdef BL_USE_MPI
  MPI_Finalize();
#endif
}

