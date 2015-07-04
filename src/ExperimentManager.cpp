#include <iostream>
#include <fstream>
#include <ExperimentManager.H>
#include <Rand.H>
#include <ParmParse.H>
#include <ParallelDescriptor.H>
#include <Utility.H>

static bool log_failed_cases_DEF = true;
static std::string log_folder_name_DEF = "FAILED";
  
ExperimentManager::ExperimentManager(ParameterManager& pmgr, ChemDriver& cd, bool _use_synthetic_data)
  : use_synthetic_data(_use_synthetic_data), verbose(true),
    parameter_manager(pmgr), expts(PArrayManage), perturbed_data(0),
    log_failed_cases(log_failed_cases_DEF), log_folder_name(log_folder_name_DEF),
    parallel_mode(PARALLELIZE_OVER_RANK)
{
  ParmParse pp;

  pp.query("log_failed_cases",log_failed_cases);
  pp.query("log_folder_name",log_folder_name);

  int nExpts = pp.countval("experiments");
  Array<std::string> experiments;
  pp.getarr("experiments",experiments,0,nExpts);
  pp.query("verbose",verbose);

  for (int i=0; i<nExpts; ++i) {
    std::string prefix = experiments[i];
    ParmParse ppe(prefix.c_str());
    std::string type; ppe.get("type",type);
    if (type == "CVReactor") {
      ZeroDReactor *cv_reactor = new ZeroDReactor(cd,experiments[i],ZeroDReactor::CONSTANT_VOLUME);
      AddExperiment(cv_reactor,experiments[i]);
    }
    else if (type == "CPReactor") {
      ZeroDReactor *cp_reactor = new ZeroDReactor(cd,experiments[i],ZeroDReactor::CONSTANT_PRESSURE);
      AddExperiment(cp_reactor,experiments[i]);
    }
    else if (type == "PREMIXReactor") {
      PREMIXReactor *premix_reactor = new PREMIXReactor(cd,experiments[i]);
      AddExperiment(premix_reactor,experiments[i]);
    }
    else {
      BoxLib::Abort("Unknown experiment type");
    }

    // Get experiment data, if not using synthetic data
    if (!use_synthetic_data) {
      BL_ASSERT(expts.defined(i));
      const SimulatedExperiment& expt = expts[i];
      int n = expt.NumMeasuredValues();
      int offset = data_offsets[i];
      true_data.resize(offset + n);

      int nd = ppe.countval("data");
      if (n < nd) {
        if (ParallelDescriptor::IOProcessor()) {
          std::cout << "Insufficient data for experiment: "
                    << prefix << ", required number data: " <<  n  << std::endl;
        }
        BoxLib::Abort();
      }
      Array<Real> tarr(n,0);
      ppe.getarr("data",tarr,0,n);
      for (int j=0; j<n; ++j) {
        true_data[offset+j] = tarr[j];
      }
    }
  }
}

std::string
ExperimentManager::GetParallelModeString() const
{
  return (parallel_mode == PARALLELIZE_OVER_RANK ? "PARALLELIZE_OVER_RANK" : "PARALLELIZE_OVER_THREAD");
}

int
ExperimentManager::NumExptData() const
{
  return num_expt_data;
}

const std::vector<Real>&
ExperimentManager::TrueData() const
{
  return true_data;
}

const std::vector<Real>&
ExperimentManager::ObservationSTD() const
{
  return true_std;
}

const std::vector<Real>&
ExperimentManager::TrueDataWithObservationNoise() const
{
  return perturbed_data;
}

void
ExperimentManager::Clear()
{
  expts.clear(); expts.resize(0, PArrayManage);
  raw_data.clear();
  data_offsets.clear();
  num_expt_data = 0;
  expt_map.clear();
}

void
ExperimentManager::AddExperiment(SimulatedExperiment* expt,
                                 const std::string&   expt_id)
{
  int num_expts_old = expts.size();
  expts.resize(num_expts_old+1,PArrayManage);
  expts.set(num_expts_old, expt);
  int num_new_values = expts[num_expts_old].NumMeasuredValues();
  raw_data.resize(num_expts_old+1,std::vector<Real>(num_new_values));
  data_offsets.resize(num_expts_old+1);
  data_offsets[num_expts_old] = ( num_expts_old == 0 ? 0 : 
                                  data_offsets[num_expts_old-1]+raw_data[num_expts_old-1].size() );
  num_expt_data = data_offsets[num_expts_old] + num_new_values;
  expt_map[expt_id] = num_expts_old;
  expt_name.push_back(expt_id);
}

void
ExperimentManager::InitializeExperiments()
{
  for (int i=0; i<expts.size(); ++i) {
    expts[i].InitializeExperiment();
  }
}

void
ExperimentManager::InitializeTrueData(const std::vector<Real>& true_parameters)
{
  if (use_synthetic_data) {
    GenerateTestMeasurements(true_parameters,true_data);
  }

  true_std.resize(NumExptData());
  for (int i=0; i<expts.size(); ++i) {

    BL_ASSERT(expts.defined(i));
    const SimulatedExperiment& expt = expts[i];
    int n = expt.NumMeasuredValues();
    BL_ASSERT(n <= raw_data[i].size());

    if (use_synthetic_data) {
      std::pair<bool,int> retVal = expts[i].GetMeasurements(raw_data[i]);
      if (!retVal.first) {
        std::string msg = SimulatedExperiment::ErrorString(retVal.second);
        std::cout << "Experiment " << i << "(" << expt_name[i]
                  << ") failed while computing synthetic data.  Error message: \""
                  << msg << "\"" << std::endl;
        BoxLib::Abort();
      }
    }

    int offset = data_offsets[i];
    int nd = raw_data[i].size();

    if (use_synthetic_data) {
      for (int j=0; j<nd; ++j) {
        true_data[offset + j] = raw_data[i][j];
      }
    }

    expts[i].GetMeasurementError(raw_data[i]);
    for (int j=0; j<nd; ++j) {
      true_std[offset + j] = raw_data[i][j];
    }
  }    
  perturbed_data.resize(0);
}

void
ExperimentManager::GenerateExptData()
{
  BL_ASSERT(perturbed_data.size()==0);
  perturbed_data.resize(num_expt_data);
  BL_ASSERT(true_std.size() == num_expt_data);
  BL_ASSERT(true_data.size() == num_expt_data);
  // FIXME: Make more general

  Real mult = 1;
  if (ParallelDescriptor::IOProcessor()) {
    //std::cout << "***************** WARNING: ZEROING DATA NOISE!!!!" << std::endl;
  }
  mult = 0;
  for(int ii=0; ii<num_expt_data; ii++){

    // FIXME: WHY IS THIS THE CORRECT THING????  (Looks wrong)
    Real small = true_std[ii];
    perturbed_data[ii] = std::max(small,true_data[ii] + true_std[ii] * randn() * mult);
  }
}

bool
ExperimentManager::EvaluateMeasurements_threaded(const std::vector<Real>& test_params,
						 std::vector<Real>&       test_measurements)
{
  bool ok = true;
  bool pvtok = ok;
  int N = expts.size();
  Array<int> msgID(N,-1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) firstprivate(pvtok)
#endif
  for (int i=0; i<N; ++i) {
    if (pvtok) {
      std::pair<bool,int> retVal = expts[i].GetMeasurements(raw_data[i]);
      if (!retVal.first) {

	msgID[i] = retVal.second;

#ifdef _OPENMP
#pragma omp critical (exp_failed)
#endif
	std::cout << "Experiment " << i << " failed.  Err msg: \""
		  << SimulatedExperiment::ErrorString(retVal.second) << "\""<< std::endl;
      }

#ifdef _OPENMP
#pragma omp atomic capture
#endif
      pvtok = ok &= retVal.first;

      int offset = data_offsets[i];
      for (int j=0, n=expts[i].NumMeasuredValues(); j<n && ok; ++j) {
	test_measurements[offset + j] = raw_data[i][j];
      }
    }
  }

  if (ok) {
    if (verbose) {
      for (int i=0; i<expts.size(); ++i) {
	int offset = data_offsets[i];
	std::cout << "Experiment " << i << " (" << expt_name[i]
		  << ") result: " << test_measurements[offset] << std::endl;
      }
    }
  }
  else {
    if (log_failed_cases) {
      LogFailedCases(test_params,test_measurements,msgID);
    }
  }

  return ok;
}

bool
ExperimentManager::EvaluateMeasurements_masterSlave(const std::vector<Real>& test_params,
						    std::vector<Real>&       test_measurements)
{
  bool ok = true;

#ifdef BL_USE_MPI
  int intok = -1;
  Array<int> msgID(expts.size(),-1);

  // All ranks use the parameters installed in the root
  ParallelDescriptor::Bcast(const_cast<Real*>(&test_params[0]), test_params.size(), 0);

  // Task parallel option over experiments - serial option follows below
  if (ParallelDescriptor::IOProcessor() && verbose){
    std::cout << "Have " << ParallelDescriptor::NProcs() << " procs " << std::endl;
  }
  bool am_worker = false;
  int master = 0;
  if (ParallelDescriptor::MyProc() == master) {
    am_worker = false;
  }
  else {
    am_worker = true;
  }
  int first_worker = 1;
  int last_worker = ParallelDescriptor::NProcs() - 1;

  typedef enum { READY, HAVE_RESULTS } workerstatus_t;
  typedef enum { WORK, STOP } workercommand_t;
  const int control_tag = 0;
  const int data_tag = 1;
  const int extra_tag = 2;

  MPI_Comm wcomm = ParallelDescriptor::Communicator();

  if (am_worker) {
    bool more_work = true;

    // Workers sit in a loop that goes: send ready, get command, act on command, send
    // ready again
    do {
      workerstatus_t mystatus; 
      workercommand_t mycommand;

      // Send signal that worker is ready to do work
      mystatus = READY;
      MPI_Send(&mystatus, 1, MPI_INTEGER, master, control_tag, wcomm);

      // Get command from master
      MPI_Recv(&mycommand,1, MPI_INTEGER, master, control_tag, wcomm, MPI_STATUS_IGNORE);

      // Act on commands
      if (mycommand==STOP) {
	more_work = false;
      }

      else if (mycommand==WORK) {

	// After command to work needs to come instructions on what to do
	int which_experiment = -1;
	ParallelDescriptor::Recv(&which_experiment,1,master,data_tag);

	if (verbose) {
	  std::cout << " Worker " << ParallelDescriptor::MyProc() << 
	    " starting on experiment number " << which_experiment <<
	    " (" << ExperimentNames() [which_experiment] << ")" << std::endl;
	}
	expts[which_experiment].CopyData(master,ParallelDescriptor::MyProc(),extra_tag);

	// Do the work
	std::pair<bool,int> retVal = expts[which_experiment].GetMeasurements(raw_data[which_experiment]);
	if (retVal.first) {
	  intok = 1;
	}
	else {
	  intok = -1;
	}
	if (verbose) {
	  std::cout << " Worker " << ParallelDescriptor::MyProc() << 
	    " finished experiment number " << which_experiment << std::endl;
	}
	// Send back the result
	mystatus = HAVE_RESULTS;
	MPI_Send(&mystatus, 1, MPI_INTEGER, master, control_tag, wcomm);
        
	ParallelDescriptor::Send(&which_experiment,1,master,data_tag);
	ParallelDescriptor::Send(&intok, 1, master, data_tag);
	ParallelDescriptor::Send(&retVal.second, 1, master, data_tag);
	ParallelDescriptor::Send(raw_data[which_experiment], master, data_tag);
	expts[which_experiment].CopyData(ParallelDescriptor::MyProc(),master,extra_tag);
	if (verbose) {
	  std::cout << " Worker " << ParallelDescriptor::MyProc() << 
	    " finished sending data back " << which_experiment << std::endl;
	}
      }
      else {
	BoxLib::Abort("Unknown command recvd");
      }
    } while (more_work);

  }
  // Master rank sits in a loop and sends out work until all of the tasks are 
  // done
  else {
    workerstatus_t worker_status; 
    workercommand_t worker_command;
    int current_worker = -1;

    int Nexperiments_dispatched = 0;
    int Nexperiments_finished = 0;

    do {
      // Look for a message from a worker  
      MPI_Status status;
      int ret = MPI_Probe(MPI_ANY_SOURCE, control_tag, wcomm, &status);
      if (ret != MPI_SUCCESS) {
	std::cout << "MPI_Probe failure " << ret << "(";
	if (ret == MPI_ERR_COMM) {
	  std::cout << "Invalid communicator";
	}
	else if (ret == MPI_ERR_TAG) {
	  std::cout << "Invalid tag";
	}
	else if (ret == MPI_ERR_RANK) {
	  std::cout << "Invalid rank";
	} else {
	  std::cout << "unknown";
	}
	std::cout << ")" << std::endl;
      }
      current_worker = status.MPI_SOURCE;
      MPI_Recv(&worker_status, 1, MPI_INTEGER, current_worker, control_tag, 
	       wcomm, MPI_STATUS_IGNORE);

      if (worker_status == READY) {
	worker_command = WORK;
	MPI_Send(&worker_command, 1, MPI_INTEGER, current_worker, control_tag, wcomm);

	// Delegate next experiment to this worker
	ParallelDescriptor::Send(&Nexperiments_dispatched,1,current_worker,data_tag);
	expts[Nexperiments_dispatched].CopyData(master,current_worker,extra_tag);

	Nexperiments_dispatched++;

      }
      else if (worker_status == HAVE_RESULTS) {
	// Fetch the results
	int exp_num;
	ParallelDescriptor::Recv(&exp_num,1,current_worker,data_tag);
	ParallelDescriptor::Recv( &intok, 1, current_worker, data_tag );
	ParallelDescriptor::Recv( &msgID[exp_num], 1, current_worker, data_tag );

	if (intok < 0) {
	  std::cout << "Experiment " << exp_num
		    << " (" << ExperimentNames() [exp_num]
		    << ") failed! Err msg: \"" << SimulatedExperiment::ErrorString(msgID[exp_num]) << "\"" << std::endl;
	  ok = false;
	}

	int n = expts[exp_num].NumMeasuredValues();
	ParallelDescriptor::Recv( raw_data[exp_num],  current_worker, data_tag );
	expts[exp_num].CopyData(current_worker,master, extra_tag);

	// Use local data about where the results go to copy the output into the 
	// test_measurements array
	int offset = data_offsets[exp_num];
	for (int j=0; j<n && (intok==1); ++j) {
	  test_measurements[offset + j] = raw_data[exp_num][j];
	}
	Nexperiments_finished++;

      } else {
	BoxLib::Abort("Unknown status from worker");
      }

    } while (Nexperiments_dispatched < expts.size() && ok);


    // All tasks sent out at this point - tell all workers to stop, getting
    // final set of results if necessary
    for (int i=first_worker; i<=last_worker; i++) {

      MPI_Recv(&worker_status, 1, MPI_INTEGER, i, control_tag, wcomm, MPI_STATUS_IGNORE);

      if (worker_status == READY) {
	worker_command = STOP;
	MPI_Send(&worker_command, 1, MPI_INTEGER, i, control_tag, wcomm);
      } 
      else if(worker_status == HAVE_RESULTS) {
	// Deal with the results, then get - hopefully - "READY" and tell worker to stop
	int exp_num;

	ParallelDescriptor::Recv(&exp_num,1,i,data_tag);
	ParallelDescriptor::Recv( &intok, 1, i, data_tag );
	ParallelDescriptor::Recv( &msgID[exp_num], 1, i, data_tag );


	if (intok < 0) {
	  std::cout << "Experiment " << exp_num
		    << " (" << ExperimentNames() [exp_num]
		    << ") failed! Err msg: \"" << SimulatedExperiment::ErrorString(msgID[exp_num]) << "\"" << std::endl;
	  ok = false;
	}

	int n = expts[exp_num].NumMeasuredValues();
	ParallelDescriptor::Recv( raw_data[exp_num],  i, data_tag );
	expts[exp_num].CopyData(i,master, extra_tag);
	int offset = data_offsets[exp_num];

	for (int j=0; j<n && (intok==1); ++j) {
	  test_measurements[offset + j] = raw_data[exp_num][j];
	}
	Nexperiments_finished++;

	MPI_Recv(&worker_status, 1, MPI_INTEGER, i, control_tag, wcomm, MPI_STATUS_IGNORE);
	worker_command = STOP;
	MPI_Send(&worker_command, 1, MPI_INTEGER, i, control_tag, wcomm);
      } 
      else { 
	BoxLib::Abort("Bad status from worker on cleanup loop");
      }
    }

    // Done. 

    if (verbose) {
      std::cout << "Sent out work for " << Nexperiments_dispatched 
		<< " experiments and had " << Nexperiments_finished 
		<< " of them done " << std::endl;
    }
    //if (Nexperiments_dispatched != Nexperiments_finished) {
    //    BoxLib::Abort("Not all dispatched experiments returned");
    //}

  }

  ParallelDescriptor::Barrier();
  // All ranks should have the same result as at root to ensure they take a reasonable
  // path through sample space when driven by an external sampler
  ParallelDescriptor::Bcast(const_cast<Real*>(&test_measurements[0]), 
			    test_measurements.size(), 0);

  ParallelDescriptor::ReduceBoolAnd(ok);

  if (ParallelDescriptor::MyProc() == master)
  {
    if (!ok) {
      if (log_failed_cases) {
	LogFailedCases(test_params,test_measurements,msgID);
      }
    }
    else {
      if (verbose) {
	for (int i=0; i<expts.size(); ++i) {
	  int offset = data_offsets[i];
	  std::cout << "Experiment " << i << " (" << expt_name[i]
		    << ") result: " << test_measurements[offset] << std::endl;
	}
      }
    }
  }

#else

  BoxLib::Abort("Master-Slave evaluation not implemented for non-MPI build");

#endif

  return ok;
}

bool
ExperimentManager::EvaluateMeasurements_parallel(const std::vector<Real>& test_params,
						 std::vector<Real>&       test_measurements)
{
  bool ok;

  if (ParallelDescriptor::NProcs() == 1) {
    ok = EvaluateMeasurements_threaded(test_params, test_measurements);
  }
  else {
    ok = EvaluateMeasurements_masterSlave(test_params, test_measurements);
  }

  return ok;
}

bool
ExperimentManager::GenerateTestMeasurements(const std::vector<Real>& test_params,
                                            std::vector<Real>&       test_measurements)
{
  for (int i=0; i<test_params.size(); ++i) {
    parameter_manager.SetParameter(i,test_params[i]);      
    if (verbose && ParallelDescriptor::IOProcessor() ){
       std::cout <<  "parameter " << i << " value " << test_params[i] << std::endl;
     }

  }
  test_measurements.resize(NumExptData());

  bool ok = true;
  if (parallel_mode == PARALLELIZE_OVER_RANK) {

    ok = EvaluateMeasurements_parallel(test_params, test_measurements);

  } else { // PARALLELIZE_OVER_THREAD

    ok = EvaluateMeasurements_threaded(test_params, test_measurements);
  }

  return ok;
}

static int failure_number = 0;
void
ExperimentManager::LogFailedCases(const std::vector<Real>& test_params,
                                  const std::vector<Real>& test_measurements,
                                  const std::vector<int>&  msgID)
{
  failure_number++;

  BL_ASSERT(ParallelDescriptor::IOProcessor());
  if (!BoxLib::UtilCreateDirectory(log_folder_name, 0755))
    BoxLib::CreateDirectoryFailed(log_folder_name);

  // First log entry in ascii table
  std::string slog_file_name = log_folder_name + "/" + "FAILURE_LOG.txt";
  std::ofstream sofs;
  sofs.open(slog_file_name.c_str(), std::ios::out | std::ios::app);
  sofs << failure_number << " ";
  for (int i=0; i<msgID.size(); ++i) {
    const std::string& msg = SimulatedExperiment::ErrorString(msgID[i]);
    if (msg != "SUCCESS" && msg != "UNKNOWN") {
      sofs << expt_name[i] << ":" << msg << " "; 
    }
  }
  sofs << '\n';
  sofs.close();

  int ndigits = std::log10(failure_number) + 1;
  const IntVect ivz(D_DECL(0,0,0));

  int np = test_params.size();
  FArrayBox p(Box(ivz,ivz),np);
  for (int i=0; i<np; ++i) {
    p(ivz,i) = test_params[i];
  }

  std::string plog_file_name = log_folder_name + "/" + "params";
  plog_file_name = BoxLib::Concatenate(plog_file_name,failure_number,ndigits);
  std::ofstream pofs; pofs.open(plog_file_name.c_str());
  p.writeOn(pofs);
  pofs.close();

  int nm = test_measurements.size();
  FArrayBox m(Box(ivz,ivz),nm);
  for (int i=0; i<nm; ++i) {
    m(ivz,i) = test_measurements[i];
  }

  std::string mlog_file_name = log_folder_name + "/" + "data";
  mlog_file_name = BoxLib::Concatenate(mlog_file_name,failure_number,ndigits);
  std::ofstream mofs; mofs.open(mlog_file_name.c_str());
  m.writeOn(mofs);
  mofs.close();
}

Real
ExperimentManager::ComputeLikelihood(const std::vector<Real>& test_data) const
{
  BL_ASSERT(test_data.size() == num_expt_data);
  if (perturbed_data.size()==0) {
    BoxLib::Abort("Must generate (perturbed) expt data before computing likelihood");
  }
  Real L = 0;
  for (int ii=0; ii<num_expt_data; ii++) {
    Real n = perturbed_data[ii] - test_data[ii];
    L += 0.5 * n * n / (true_std[ii] * true_std[ii]);
  }
  return L;
}


bool 
ExperimentManager::isgoodParamVal( Real k, std::vector<Real> & pvals, int idx ){
  int num_dvals = NumExptData();
  std::vector<Real> dvals(num_dvals);
  pvals[idx] = k;
  GenerateTestMeasurements(pvals, dvals);
  //std::cout << "trying p (" << idx << ") = " << k << " dval[0] = " << dvals[0] << std::endl;
  bool res = true;
  for(int id=0; id<num_dvals; id++ ){
      if ( dvals[id] < 0.0 ) res = false;
  }
  return res;

}

void 
ExperimentManager::get_param_limits( Real * kmin, Real * kmax, Real * ktyp, Real tol, 
                       std::vector<Real> & pvals, int idx){

    double k1, k2, ktest, delt;

    // First check right hand value - don't bother if it's ok
    if( !isgoodParamVal( *kmax, pvals, idx) ) {
        k2 = *kmax;
        k1 = *ktyp;
        do {
            ktest = (k2+k1)*0.5;
            if( isgoodParamVal( ktest, pvals, idx) ){
                k1 = ktest;
            }
            else{
                k2 = ktest;
            }
            delt = (k2-k1);
        } while( delt > tol);
        *kmax = k1;
    }

    if( !isgoodParamVal( *kmin, pvals, idx ) ){
        k1 = *kmin;
        k2 = *ktyp;
        do {
            ktest = (k2+k1)*0.5;
            if( isgoodParamVal( ktest, pvals, idx) ){
                k2 = ktest;
            }
            else{
                k1 = ktest;
            }
            delt = (k2-k1);
        } while( delt > tol);
        *kmin = k2;

    }

}

void 
ExperimentManager::get_param_interesting( Real * kmin, Real * kmax, Real * ktyp, Real tol, 
                                          std::vector<Real> & pvals, int idx){

    double k1, k2, ktest, delt;

    // First check right hand value - don't bother if it's ok
    if( !isgoodParamVal( *kmax, pvals, idx) ) {
        k2 = *kmax;
        k1 = *ktyp;
        do {
            ktest = (k2+k1)*0.5;
            if( isgoodParamVal( ktest, pvals, idx) ){
                k1 = ktest;
            }
            else{
                k2 = ktest;
            }
            delt = (k2-k1);
        } while( delt > tol);
        *kmax = k1;
    }

        {
            double dlast, dmag;
            int num_dvals = NumExptData();
            std::vector<Real> dvals(num_dvals);
            pvals[idx] = *kmax;
            GenerateTestMeasurements(pvals, dvals);
            dlast = dmag = dvals[0];
            k1 = *kmax;
            tol = dmag*0.1;
            double dk = *kmax*0.01;
            std::cout << " looking for change bigger than : " << tol << std::endl;
            do {
                int num_dvals = NumExptData();
                std::vector<Real> dvals(num_dvals);
                pvals[idx] = k1 - dk;
                GenerateTestMeasurements(pvals, dvals);

                delt = fabs( dlast - dvals[0] );
                dlast = dvals[0];
                if( delt < tol ) k1 = k1 -dk;
                std::cout << " k1, dlast: " << k1 << "; " << dlast << std::endl;

            } while( delt < tol );
        }
        *kmax = k1;

        // Start from kmax and shrink until just before interesting chanage


    if( !isgoodParamVal( *kmin, pvals, idx ) ){
        k1 = *kmin;
        k2 = *ktyp;
        do {
            ktest = (k2+k1)*0.5;
            if( isgoodParamVal( ktest, pvals, idx) ){
                k2 = ktest;
            }
            else{
                k1 = ktest;
            }
            delt = (k2-k1);
        } while( delt > tol);
        *kmin = k2;

    }

}
