#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>
#include <Utility.H>

#include <iostream>
#include <iomanip>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <PremixSol.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

static void GATHER_TO_IO(std::vector<Real>& send_data,
			 int                 num_send,
			 std::vector<Real>& recv_data,
			 int&                num_recv)
{
  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();

  std::vector<int> recv_cnts(nprocs,0);
  std::vector<int> rdispls(nprocs,0);

  recv_cnts[ParallelDescriptor::MyProc()] = num_send;
  ParallelDescriptor::ReduceIntSum(&(recv_cnts[0]),nprocs);

  num_recv = 0;
  if (ParallelDescriptor::IOProcessor()) {
    for (int i=0; i<nprocs; ++i) {
      num_recv += recv_cnts[i];
      if (i == 0) {
	rdispls[0] = 0;
      }
      else {
	rdispls[i] = rdispls[i-1] + recv_cnts[i-1];
      }
    }
    recv_data.resize(num_recv);
  }
#if BL_USE_MPI
  BL_MPI_REQUIRE( MPI_Gatherv(num_send == 0 ? 0 : &(send_data[0]),
			      num_send,
			      ParallelDescriptor::Mpi_typemap<Real>::type(),
			      num_recv == 0 ? 0 : &(recv_data[0]),
			      &(recv_cnts[0]),
			      &(rdispls[0]),
			      ParallelDescriptor::Mpi_typemap<Real>::type(),
			      ParallelDescriptor::IOProcessorNumber(),
			      ParallelDescriptor::Communicator()) );
#endif
}

std::vector<Real>
GetBoundedSample(const std::vector<Real>& prior_mean,
		 const std::vector<Real>& prior_std,
		 const std::vector<Real>& upper_bound,
		 const std::vector<Real>& lower_bound)
{
  int N = prior_mean.size();
  std::vector<Real> sample(N);
  for (int i=0; i<N; ++i) {
    Real r = BoxLib::Random() - 0.5;
    sample[i] = prior_mean[i] + prior_std[i]*r;
    bool sample_oob = (sample[i] < lower_bound[i] || sample[i] > upper_bound[i]);	
    while (sample_oob) {
      sample[i] = prior_mean[i] + prior_std[i]*BoxLib::Random();
      sample_oob = (sample[i] < lower_bound[i] || sample[i] > upper_bound[i]);
    }
  }
  return sample;
}

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

  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();


  ParmParse pp;

  ParameterManager& parameter_manager = driver.mystruct->parameter_manager;
  ExperimentManager& expt_manager = driver.mystruct->expt_manager;
  expt_manager.SetVerbose(false);
  expt_manager.SetParallelMode(ExperimentManager::PARALLELIZE_OVER_THREAD);

  const std::vector<Real>& prior_mean = parameter_manager.PriorMean();
  const std::vector<Real>& ensemble_std = parameter_manager.EnsembleSTD();
  const std::vector<Real>& upper_bound = parameter_manager.UpperBound();
  const std::vector<Real>& lower_bound = parameter_manager.LowerBound();
  int num_params = ensemble_std.size();

  Minimizer* minimizer = 0;
  std::string which_minimizer = "nlls";
  pp.query("which_minimizer",which_minimizer);
  bool hasHessian;
  if (which_minimizer == "nlls") {
    minimizer = new NLLSMinimizer();
    hasHessian = true;
  }
  else {
    minimizer = new GeneralMinimizer();
    hasHessian = false;
  }

  int NOS = 1; pp.query("NOS",NOS);
  int max_failures = 5; pp.query("max_failures",max_failures);
  std::vector<Real> samples(NOS * num_params);
  std::vector<Real> solns(NOS * num_params);
  std::vector<Real> F(NOS);
  std::vector<Real> hessians;
  if (hasHessian) {
    hessians.resize(NOS * num_params * num_params);
  }

  int num_failures = 0;
  std::vector<Real> thisSolns(num_params);

  int j;
  for (j=0; j<NOS && num_failures<max_failures; ++j)
  {
    bool ok = false;
    std::vector<Real> thisSample(num_params);
    while (ok == false && num_failures<max_failures) {
      thisSample = GetBoundedSample(prior_mean, ensemble_std, upper_bound, lower_bound);
      ok = minimizer->minimize((void*)(driver.mystruct), thisSample, thisSolns);
      if (!ok) {
	num_failures++;
      } else {
	F[j] = driver.LogLikelihood(thisSolns);
      }
    }

    long offset = j * num_params;
    for (int i=0; i<num_params; ++i) {
      samples[offset+i] = thisSample[i];
      solns[offset+i] = thisSolns[i];
    }

    if (hasHessian) {
      NLLSMinimizer* nm = dynamic_cast<NLLSMinimizer*>(minimizer);
      BL_ASSERT(nm!=0);
      MyMat H = nm->JTJ((void*)(driver.mystruct),thisSolns);

      offset = j * num_params * num_params;
      for (int i=0; i<num_params; ++i) {
	for (int k=0; k<num_params; ++k) {
	  hessians[offset + i*num_params + k] = H[i][k];
	}
      }
    }
  }
  int NOSloc = j;

  std::vector<Real> samplesg;
  std::vector<Real> solnsg;
  int num_send = num_params * NOSloc;
  int num_recv = num_send;

  GATHER_TO_IO(  solns, num_send, solnsg,   num_recv); solns.clear();
  GATHER_TO_IO(samples, num_send, samplesg, num_recv); samples.clear();

  std::vector<Real> Fg;
  int num_send_F = NOSloc;
  int num_recv_F = num_send;
  GATHER_TO_IO(F, num_send_F, Fg, num_recv_F); F.clear();

  std::vector<Real> hessiansg;
  int num_send_h, num_recv_h;
  if (hasHessian) {
    num_send_h = num_params * num_params * NOSloc;
    num_recv_h = num_send_h;
    GATHER_TO_IO(hessians, num_send_h, hessiansg, num_recv_h); hessians.clear();
  }

  std::string samples_file = "samples"; pp.query("samples",samples_file);
  std::string solns_file = "solns"; pp.query("solns",solns_file);
  std::string Fs_file = "Fs"; pp.query("Fs",Fs_file);
  std::string hessians_file = "hessians"; pp.query("hessians",hessians_file);

  if (myproc == 0) {

    int NOStot = num_recv / num_params;
    BL_ASSERT(NOStot * num_params == num_recv);

    // Transpose sample data to be compatible with pltfile format
    std::vector<Real> samplesT(num_params * NOStot);
    std::vector<Real> solnsT(num_params * NOStot);
    for (int k=0; k<NOStot; ++k) {	
      for (int i=0; i<num_params; ++i) {
	samplesT[k + i*NOStot] = samplesg[i + k*num_params];
	solnsT[  k + i*NOStot] = solnsg[  i + k*num_params];
      }
    }
    samplesg.clear();
    solnsg.clear();

    UqPlotfile pfi(samplesT,num_params,1,0,NOStot,"");
    pfi.Write(samples_file);
    samplesT.clear();

    UqPlotfile pfo(solnsT,num_params,1,0,NOStot,"");
    pfo.Write(solns_file);
    solnsT.clear();

    UqPlotfile pff(Fg,1,1,0,NOStot,"");
    pff.Write(Fs_file);
    Fg.clear();

    if (hasHessian) {
      std::vector<Real> hessiansT(num_params * num_params * NOStot);
      for (int j=0; j<NOStot; ++j) {
	for (int i=0; i<num_params; ++i) {
	  for (int k=0; k<num_params; ++k) {
	    hessiansT[NOStot*(i*num_params + k) + j] = hessiansg[j*num_params*num_params + i*num_params + k];
	  }
	}
      }
      hessiansg.clear();

      UqPlotfile pfH(hessiansT,num_params*num_params,1,0,NOStot,"");
      pfH.Write(hessians_file);
      hessiansT.clear();
    }
  }

  delete minimizer;

  BoxLib::Finalize();

}

