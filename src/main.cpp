#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>

#include <iostream>
#include <fstream>

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
  Driver driver(argc,argv,1);
  driver.SetComm(MPI_COMM_WORLD);
  driver.init(argc,argv);
#else
  Driver driver(argc,argv,0);
#endif

  bool ioproc = ParallelDescriptor::IOProcessor();

  ParmParse pp;

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  expt_manager.SetVerbose(false);
  //expt_manager.SetVerbose(true);

  std::vector<Real> guess_params;
  const std::vector<Real>& prior_std = parameter_manager.PriorSTD();
  int num_params = prior_std.size();

  const std::vector<Real>& true_data = expt_manager.TrueData();
  const std::vector<Real>& perturbed_data = expt_manager.TrueDataWithObservationNoise();
  const std::vector<Real>& true_data_std = expt_manager.ObservationSTD();
  int num_data = true_data.size();

  if (pp.countval("initFile")) {

    std::string initFile; pp.get("initFile",initFile);
    UqPlotfile pf;
    pf.Read(initFile);
    int iter = pf.ITER() + pf.NITERS() - 1;
    int iters = 1;
    int initID = iter; pp.query("initID",initID);
    if (initID < pf.ITER() || initID >= iter+iters) {
      BoxLib::Abort("InitID sample not in initFile");
    }
    guess_params = pf.LoadEnsemble(initID, iters);

    if (ioproc) {
      for(int ii=0; ii<num_params; ii++){
	std::cout << "Guess (" << ii << "): " << guess_params[ii] << std::endl;
      }
    }

  } else {

    const std::vector<Real>& true_params = parameter_manager.TrueParameters();
    const std::vector<Real>& prior_mean = parameter_manager.PriorMean();
    num_params = true_params.size();

    Real ic_pert=0; pp.query("ic_pert",ic_pert);
    guess_params.resize(num_params);
    if (ic_pert == 0) {
      for(int i=0; i<num_params; i++){
	guess_params[i] = prior_mean[i];
      }
    }
    else {
      const std::vector<Real>& upper_bound = parameter_manager.UpperBound();
      const std::vector<Real>& lower_bound = parameter_manager.LowerBound();
      for(int i=0; i<num_params; i++){
	guess_params[i] = prior_mean[i] + prior_std[i]*randn()*ic_pert;
	bool sample_oob = (guess_params[i] < lower_bound[i] || guess_params[i] > upper_bound[i]);
	
	while (sample_oob) {
	  if (ioproc) {
	    std::cout <<  "sample is out of bounds, parameter " << i
		      << " val,lb,ub: " << guess_params[i]
		      << ", " << lower_bound[i] << ", " << upper_bound[i] << std::endl;
	  }
	  guess_params[i] = prior_mean[i] + prior_std[i]*randn();
	  sample_oob = (guess_params[i] < lower_bound[i] || guess_params[i] > upper_bound[i]);
	}
      }
    }
    ParallelDescriptor::Bcast(&(guess_params[0]),guess_params.size(),ParallelDescriptor::IOProcessorNumber());

    if (pp.countval("init_samples_file") > 0) {
      std::string init_samples_file; pp.get("init_samples_file",init_samples_file);
      if (ioproc) {
	std::cout << "Writing initial samples to: " << init_samples_file << std::endl;
      }
      UqPlotfile pf(guess_params,num_params,1,0,1,"");
      pf.Write(init_samples_file);
    }

    if (pp.countval("init_samples_file") > 0 ) {
      std::string init_samples_file; pp.get("init_samples_file",init_samples_file);
      if (ioproc) {
	std::cout << "Writing initial samples to: " << init_samples_file << std::endl;
      }
      UqPlotfile pf(guess_params,num_params,1,0,1,"");
      pf.Write(init_samples_file);
    }

    bool show_initial_stats = false; pp.query("show_initial_stats",show_initial_stats);
    if (show_initial_stats && ioproc) {
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
      Real F = NegativeLogLikelihood(prior_mean);

      if (ioproc) {
	std::cout << "F at true parameters = " << Ftrue << std::endl;
	std::cout << "F at prior mean = " << F << std::endl;
      }
    }
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
  bool do_minimize = which_minimizer != "none" && which_sampler != "prior_mc";
  pp.query("do_minimize",do_minimize);

  if (!do_minimize) {

    UqPlotfile pf;
    pf.Read(samples_at_min_file);
    int iter = pf.ITER() + pf.NITERS() - 1;
    int iters = 1;
    soln_params = pf.LoadEnsemble(iter, iters);
    ParallelDescriptor::Bcast(&(soln_params[0]),soln_params.size(),ParallelDescriptor::IOProcessorNumber());

  } else {

    if (ioproc) {
      std::cout << "Doing minimization..." << std::endl;
    }
    if (which_minimizer == "nlls") {
      minimizer = new NLLSMinimizer();
    }
    else {
      minimizer = new GeneralMinimizer();
    }

    minimizer->minimize((void*)(driver.mystruct), guess_params, soln_params);

    if (ioproc) {
      std::cout << "Final parameters: " << std::endl;
      for(int ii=0; ii<num_params; ii++){
	std::cout << parameter_manager[ii] << std::endl;
      }
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
    sampler = new PriorMCSampler(guess_params, prior_std);
  }
  else {
    if (ioproc) {
      std::cout << "Getting Hessian... " << std::endl;
    }
    MyMat H, InvSqrtH;

    std::string hessianInFile;
    if (pp.countval("hessianInFile")) {
      pp.get("hessianInFile",hessianInFile); 
      std::cout << "      Getting Hessian from file: " << hessianInFile << std::endl;
      std::ifstream hessianIS(hessianInFile.c_str());
      H = readHessian(hessianIS);
      hessianIS.close();
    }
    else {
      if (fd_Hessian) {
	if (ioproc) {
	  std::cout << "      Getting Hessian with finite differences... " << std::endl;
	}
        H = Minimizer::FD_Hessian((void*)driver.mystruct, soln_params);
      }
      else {
	if (ioproc) {
	  std::cout << "      Getting Hessian as JTJ... " << std::endl;
	}
        int n = num_params;
        int m = num_data + n;
        std::vector<Real> FVEC(m);
        std::vector<Real> FJAC(m*n);
        NLLSMinimizer* nm = dynamic_cast<NLLSMinimizer*>(minimizer);
        BL_ASSERT(nm!=0);
        H = nm->JTJ((void*)(driver.mystruct),soln_params);
      }
    }
    std::string hessianOutFile;
    if (pp.countval("hessianOutFile")) {
      pp.get("hessianOutFile",hessianOutFile); 
      if (ioproc) {
	std::cout << "Writing Hessian to " << hessianOutFile << std::endl;
	std::ofstream hessianOS(hessianOutFile.c_str());
	writeHessian(H,hessianOS);
	hessianOS.close();
      }
    }

    InvSqrtH = Minimizer::InvSqrt((void*)driver.mystruct, H);

    if (which_sampler == "linear_map" ||
        which_sampler == "symmetrized_linear_map") {

      // Output value of objective function at minimum
      if (ioproc) {
	std::cout << "Computing logLikelihood at minimum state..." << std::endl;
      }
      Real phi = NegativeLogLikelihood(soln_params);
      if (ioproc) {
	std::cout << "F at numerical minimum = " << phi << std::endl;
      }

      if (which_sampler == "linear_map") {
        sampler = new LinearMapSampler(soln_params,H,InvSqrtH,phi);
      }
      else {
        sampler = new SymmetrizedLinearMapSampler(soln_params,H,InvSqrtH,phi);
      }
    }
    else if (which_sampler == "none") {
    }
    else {
      BoxLib::Abort("Invalid value for which_sampler");
    }
  }

  if (sampler) {
    if (ioproc) {
      std::cout << "Sampling..." << std::endl;
    }
    sampler->Sample((void*)(driver.mystruct), samples, w);
    if (ioproc) {
      std::cout << "...Finished" << std::endl;
    }
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

