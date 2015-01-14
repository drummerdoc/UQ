#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>

#include <iostream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <PremixSol.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

int
main (int   argc,
      char* argv[])
{
#ifdef BL_USE_MPI
  MPI_Init (&argc, &argv);
  Driver driver(argc,argv,MPI_COMM_WORLD);
#else
  Driver driver(argc,argv, 0);
#endif

  ParmParse pp;

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  //expt_manager.SetVerbose(false);
  expt_manager.SetVerbose(true);

  
  const std::vector<Real>& true_data = expt_manager.TrueData();
  const std::vector<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();
  const std::vector<Real>& true_data_std = expt_manager.ObservationSTD();
  int num_data = true_data.size();

  const std::vector<Real>& true_params = parameter_manager.TrueParameters();
  const std::vector<Real>& prior_mean = parameter_manager.PriorMean();
  const std::vector<Real>& prior_std = parameter_manager.PriorSTD();
  int num_params = true_params.size();

  bool show_initial_stats = false; pp.query("show_initial_stats",show_initial_stats);
  if (show_initial_stats) {
    std::cout << "True and noisy data: (npts=" << num_data << ")\n"; 
    for(int ii=0; ii<num_data; ii++){
      std::cout << "  True: " << true_data[ii]
                << "  Noisy: " << perturbed_data[ii]
                << "  Standard deviation: " << true_data_std[ii] << std::endl;
    }

    std::cout << "True and prior mean:\n"; 
    for(int ii=0; ii<num_params; ii++){
      std::cout << "  True: " << true_params[ii]
                << "  Prior: " << prior_mean[ii]
                << "  Standard deviation: " << prior_std[ii] 
                << "  Difference / std: "   <<  (true_params[ii]-prior_mean[ii])/prior_std[ii] 
                << std::endl;
    }

    Real Ftrue = NegativeLogLikelihood(true_params);
    std::cout << "F at true parameters = " << Ftrue << std::endl;
  
    Real F = NegativeLogLikelihood(prior_mean);
    std::cout << "F at prior mean = " << F << std::endl;
  }


  /*
    Minimize system (if reqd)
    =========================

    If minimized solution required (which_sampler != prior_mc), either minimize
    (which_minimizer="nlls" or != "none"), or read sample from file containing
    minimum point (which_minimizer="none").
   */
  std::string which_sampler = "prior_mc"; pp.query("which_sampler",which_sampler);
  std::vector<Real> soln_params(num_params);

  Minimizer* minimizer = 0;
  std::string samples_at_min_file = "ParamsAtMin";
  pp.query("samples_at_min_file",samples_at_min_file);

  std::string which_minimizer = "none";
  if (which_sampler != "prior_mc") {
    which_minimizer = "nlls";
  }
  pp.query("which_minimizer",which_minimizer);

  if (which_minimizer == "none" && which_sampler != "prior_mc") {

    UqPlotfile pf;
    pf.Read(samples_at_min_file);
    int iter = pf.ITER() + pf.NITERS() - 1;
    int iters = 1;
    soln_params = pf.LoadEnsemble(iter, iters);

  } else {
    if (which_minimizer == "nlls") {
      minimizer = new NLLSMinimizer();
    }
    else {
      minimizer = new GeneralMinimizer();
    }

    std::vector<Real> guess_params(num_params);
    for(int i=0; i<num_params; i++){
      guess_params[i] = prior_mean[i];
    }

    minimizer->minimize((void*)(driver.mystruct), guess_params, soln_params);

    std::cout << "Final parameters: " << std::endl;
    for(int ii=0; ii<num_params; ii++){
      std::cout << parameter_manager[ii] << std::endl;
    }

    UqPlotfile pf(soln_params,num_params,1,0,1,"");
    pf.Write(samples_at_min_file);

    return 0;
  }


  /*
    Do sampling
    ===========
   */
  int NOS = 10000; pp.query("NOS",NOS);
  std::vector<Real> w(NOS);
  std::vector<std::vector<Real> > samples(NOS, std::vector<Real>(num_params,-1));

  bool fd_Hessian = true; pp.query("fd_Hessian",fd_Hessian);

  Sampler *sampler = 0;
  if (which_sampler == "prior_mc") {
    sampler = new PriorMCSampler(prior_mean, prior_std);
  }
  else {
    // Output value of objective function at minimum
    Real phi = NegativeLogLikelihood(soln_params);
    std::cout << "F at numerical minimum = " << phi << std::endl;

    MyMat H, InvSqrtH;
    if (fd_Hessian) {
      /*
        Compute Finite Difference Hessian
      */
      H = Minimizer::FD_Hessian((void*)driver.mystruct, soln_params);
    }
    else {
      /*
        Compute J^t J
      */
      int n = num_params;
      int m = num_data + n;
      std::vector<Real> FVEC(m);
      std::vector<Real> FJAC(m*n);
      NLLSMinimizer* nm = dynamic_cast<NLLSMinimizer*>(minimizer);
      BL_ASSERT(nm!=0);
      H = nm->JTJ((void*)(driver.mystruct),soln_params);
    }

    InvSqrtH = Minimizer::InvSqrt((void*)driver.mystruct, H);

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
    std::cout << "Sampling..." << std::endl;
    sampler->Sample((void*)(driver.mystruct), samples, w);
    std::cout << "...Finished" << std::endl;

    std::string samples_outfile = "mysamples";

    int len = NOS * num_params;
    std::vector<double> samplesT(len);
    for (int i=0; i<NOS; ++i) {
      for (int j=0; j<num_params; ++j) {
        int index = i + j*NOS;
        samplesT[index] = samples[i][j];
      }
    }
    UqPlotfile pf(samplesT,num_params,1,0,NOS,"");
    pf.Write(samples_outfile);
  }
  delete sampler;
  delete minimizer;

  BoxLib::Finalize();

#ifdef BL_USE_MPI
  MPI_Finalize();
#endif
}

