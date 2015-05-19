#include <Driver.H>
#include <ChemDriver.H>
#include <Sampler.H>
#include <Utility.H>

#include <iostream>
#include <fstream>

#include <ParmParse.H>
#include <SimulatedExperiment.H>
#include <PremixSol.H>
#include <UqPlotfile.H>

#include <ParallelDescriptor.H>

static void GATHER_TO_IO(std::vector<Real>& send_data,
			 int                num_send,
			 std::vector<Real>& recv_data)
{
  int num_procs = ParallelDescriptor::NProcs();
  std::vector<int> recv_cnts(num_procs,0);
  std::vector<int> rdispls(num_procs,0);

  recv_cnts[ParallelDescriptor::MyProc()] = num_send;
  ParallelDescriptor::ReduceIntSum(&(recv_cnts[0]),num_procs);

  int tot_num_to_recv = 0;
  if (ParallelDescriptor::IOProcessor()) {
    for (int i=0; i<num_procs; ++i) {
      tot_num_to_recv += recv_cnts[i];
      if (i == 0) {
	rdispls[0] = 0;
      }
      else {
	rdispls[i] = rdispls[i-1] + recv_cnts[i-1];
      }
    }
    recv_data.resize(tot_num_to_recv);
  }
#if BL_USE_MPI
  BL_MPI_REQUIRE( MPI_Gatherv(num_send == 0 ? 0 : &(send_data[0]),
			      num_send,
			      ParallelDescriptor::Mpi_typemap<Real>::type(),
			      tot_num_to_recv == 0 ? 0 : &(recv_data[0]),
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
    sample[i] = prior_mean[i] + prior_std[i]*BoxLib::Random();
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
  if (which_minimizer == "nlls") {
    minimizer = new NLLSMinimizer();
  }
  else {
    minimizer = new GeneralMinimizer();
  }

  int NOS = 1; pp.query("NOS",NOS);
  int max_failures = 5; pp.query("max_failures",max_failures);
  std::vector<Real> samples(NOS * num_params);
  std::vector<Real> solns(NOS * num_params);

  int nprocs = ParallelDescriptor::NProcs();
  int myproc = ParallelDescriptor::MyProc();

  int num_failures = 0;
  std::vector<Real> thisSolns(num_params);


  for (int j=0; j<NOS && num_failures<max_failures; ++j)
  {
    bool ok = false;
    std::vector<Real> thisSample(num_params);
    while (ok == false && num_failures<max_failures) {
      thisSample = GetBoundedSample(prior_mean, ensemble_std, upper_bound, lower_bound);
      ok = minimizer->minimize((void*)(driver.mystruct), thisSample, thisSolns);
      if (!ok) {
	num_failures++;
      }
    }

    long offset = j * num_params;
    for (int i=0; i<num_params; ++i) {
      samples[offset+i] = thisSample[i];
      solns[offset+i] = thisSolns[i];
    }
  }

  std::vector<Real> samplesg;
  std::vector<Real> solnsg;
  GATHER_TO_IO(solns,NOS * num_params, solnsg);
  GATHER_TO_IO(samples,NOS * num_params, samplesg);

  if (myproc == 0) {
    std::string samples_file = "samples"; pp.query("samples",samples_file);
    std::string solns_file = "solns"; pp.query("solns",solns_file);

    UqPlotfile pfi(samplesg,num_params,1,0,NOS*nprocs,"");
    pfi.Write(samples_file);

    UqPlotfile pfo(solnsg,num_params,1,0,NOS*nprocs,"");
    pfo.Write(solns_file);
  }

  delete minimizer;

  BoxLib::Finalize();

}

