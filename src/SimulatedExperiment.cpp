#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <SimulatedExperiment.H>
#include <ParmParse.H>
#include <Utility.H>

#include <sys/time.h>

#include <ParallelDescriptor.H>

#ifdef _OPENMP
#include "omp.h"
#endif


static Real Patm_DEF = 1;
static Real dt_DEF   = 0.1;
static Real Tfile_DEF = -1;
static int  num_time_intervals_DEF = -1;
static std::string diagnostic_name_DEF = "temp";
static Real ZeroDReactorErr_DEF = 15;
static Real PREMIXReactorErr_DEF = 10;
static Real dpdt_thresh_DEF = 10; // atm / s
static Real dOH_thresh_DEF = 1.0e-4; // Arbitrary default
static std::string log_file_DEF = "NULL"; // if this, no log
static int verbosity_DEF = 0;

static int max_premix_iters_DEF = 100000;
static int min_reasonable_regrid_DEF = 24;
static std::string diagnostic_prefix_DEF = "VERBOSE_";

static std::string getFilePart(const std::string& path)
{
  std::vector<std::string> parts = BoxLib::Tokenize(path,"/");
  return parts[parts.size()-1];
}

static std::string getDirPart(const std::string& path)
{
  std::vector<std::string> parts = BoxLib::Tokenize(path,"/");

  std::string ret;
  if (path.at(0) == '/') {
    ret = "/";
  }
  else if (parts.size() == 1) {
    ret = "./";
  }

  for (int i=0; i<parts.size()-1; ++i) {
    ret += parts[i];
    if (i!=parts.size()-2) ret += '/';
  }
  return ret;
}

static void EnsureFolderExists(const std::string& fullPath)
{
  if (ParallelDescriptor::IOProcessor()) {
#ifdef _OPENMP
#pragma omp critical (mk_diag_folder)
#endif
    {
      std::string dirPart = getDirPart(fullPath);
      if( ! BoxLib::UtilCreateDirectory(dirPart, 0755)) {
	BoxLib::CreateDirectoryFailed(dirPart);
      }
    }
  }
  ParallelDescriptor::Barrier();
}

SimulatedExperiment::ErrMap
SimulatedExperiment::build_err_map()
{
  ErrMap my_map;
  my_map.push_back("PREREQ_FAILED");
  my_map.push_back("INVALID_OBSERVATION_1");
  my_map.push_back("INVALID_OBSERVATION_2");
  my_map.push_back("INVALID_OBSERVATION_3");
  my_map.push_back("INVALID_OBSERVATION_4");
  my_map.push_back("INVALID_OBSERVATION_5");
  my_map.push_back("INVALID_OBSERVATION_6");
  my_map.push_back("INVALID_OBSERVATION_7");
  my_map.push_back("PREMIX_TOO_MANY_ITERS");
  my_map.push_back("PREMIX_SOLVER_FAILED");
  //my_map.push_back("NEEDED_MEAN_BUT_NOT_FINISHED");
  my_map.push_back("NEEDED_MEAN_REFINE");
  my_map.push_back("NEEDED_MEAN_BUT_NOT_FINISHED");
  my_map.push_back("REACTOR_DID_NOT_COMPLETE");
  my_map.push_back("VODE_FAILED");
  my_map.push_back("SUCCESS");
  return my_map;
}
static std::string UNKNOWN = "UNKNOWN";

SimulatedExperiment::ErrMap SimulatedExperiment::err_map = SimulatedExperiment::build_err_map();

const std::string&
SimulatedExperiment::ErrorString(int errID) {
  return errID < 0 || errID >= err_map.size() ? UNKNOWN : err_map[errID];
}

int
SimulatedExperiment::ErrorID(const std::string& errStr) {
  for (int i=0; i<err_map.size(); ++i) {
    if (err_map[i] == errStr) return i;
  }
  return -1;
}

SimulatedExperiment::SimulatedExperiment()
  :  is_initialized(false), log_file(log_file_DEF),
     verbosity(verbosity_DEF),
     diagnostic_prefix(diagnostic_prefix_DEF)
{
}

SimulatedExperiment::~SimulatedExperiment() {}

void SimulatedExperiment::CopyData(int src, int dest, int tag){}

void SimulatedExperiment::SetDiagnosticFilePrefix(const std::string& prefix)
{
  diagnostic_prefix = prefix;
}

int
ZeroDReactor::NumMeasuredValues() const {return num_measured_values;}

ZeroDReactor::~ZeroDReactor() {}

const std::vector<Real>&
ZeroDReactor::GetMeasurementTimes() const
{
  return measurement_times;
}

ZeroDReactor::ZeroDReactor(ChemDriver& _cd, const std::string& pp_prefix, const REACTOR_TYPE& _type)
  : SimulatedExperiment(), name(pp_prefix), cd(_cd), reactor_type(_type),num_measured_values(0),
    sCompY(-1), sCompT(-1), sCompR(-1), sCompRH(-1)
{
  ParmParse pp(pp_prefix.c_str());

  std::string expt_type; pp.get("type",expt_type);

  pp.query("verbosity",verbosity);

  if (expt_type != "CVReactor" && expt_type != "CPReactor") {
    std::string err = "Inputs incompatible with experiment type: " + pp_prefix;
    BoxLib::Abort(err.c_str());
  }

  Real data_tstart = 0; pp.query("data_tstart",data_tstart);
  Real data_tend = dt_DEF; pp.query("data_tend",data_tend);
  int data_num_points = num_time_intervals_DEF;
  pp.query("data_num_points",data_num_points); BL_ASSERT(data_num_points>0);

  measurement_times.resize(data_num_points);
  Real dt = data_tend - data_tstart;  BL_ASSERT(dt>=0);
  for (int i=0; i<data_num_points; ++i) {
    measurement_times[i] = data_tstart + i*dt/(data_num_points-1);
  }

  //
  // Define initial state of reactor:
  //
  //  Tfile > 0:  Read pmf file, use state near T=Tfile
  //  else:
  //   require Tinit, read in volume fractions, X, of species (by name)
  //       (note, will linearly scale to sum(X) = 1
  //
  Patm = Patm_DEF; pp.get("Patm",Patm);

  // Ordering of variables in pmf file used for initial conditions
  sCompT  = 1;
  sCompRH = 2;
  sCompR  = 3;
  sCompY  = 4;

  Tfile = Tfile_DEF; pp.query("Tfile",Tfile);

  if (Tfile > 0) {
    pp.get("pmf_file_name",pmf_file_name);
  }
  else {
    int nSpec = cd.numSpecies();
    Array<Real> volFrac(nSpec,0);
    Real tot = 0;
    for (int i=0; i<nSpec; ++i) {
      const std::string& name = cd.speciesNames()[i];
      if (pp.countval(name.c_str()) > 0) {
        pp.get(name.c_str(),volFrac[i]);
        tot += volFrac[i];
      }
    }
    if (tot <=0 ) {
      BoxLib::Abort("Reactor must be initialized with at least one species");
    }
    for (int i=0; i<nSpec; ++i) {
      volFrac[i] *= 1/tot;
    }
    Real Tinit = -1; pp.get("T",Tinit);

    IntVect iv(D_DECL(0,0,0));
    Box bx(iv,iv);
    funcCnt.resize(bx,1);

    const int nComp = nSpec + 4;
    s_init.resize(bx,nComp);
    s_init(iv,sCompT) = Tinit;
	
    Array<Real> Y = cd.moleFracToMassFrac(volFrac);

    for (int i=0; i<nSpec; ++i) {
      s_init(iv,sCompY+i) = Y[i];
    }
    cd.getRhoGivenPTY(s_init,Patm,s_init,s_init,bx,sCompT,sCompY,sCompR);
  }

  measured_comps.resize(1);
  diagnostic_name = diagnostic_name_DEF;
  pp.query("diagnostic_name",diagnostic_name);
  if (diagnostic_name == "temp") {
    measured_comps[0] = sCompT;
    num_measured_values = measurement_times.size() * measured_comps.size();
  }
  else if (diagnostic_name == "pressure") {
    measured_comps[0] = -1; // Pressure
    num_measured_values = measurement_times.size() * measured_comps.size();
  }
  else if (diagnostic_name == "max_pressure") {
    measured_comps[0] = -1; // Pressure
    pp.query("p_thresh",transient_thresh);
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "pressure_rise") {
    transient_thresh = dpdt_thresh_DEF;
    pp.query("dpdt_thresh",transient_thresh);
    measured_comps[0] = -1; // Pressure
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "onset_pressure_rise") {
    transient_thresh = dpdt_thresh_DEF;
    pp.query("dpdt_thresh",transient_thresh);
    measured_comps[0] = -1; // Pressure
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "max_OH" || diagnostic_name == "inflect_OH" || diagnostic_name == "onset_OH") {
    transient_thresh = dpdt_thresh_DEF;
    pp.query("dOH_thresh",transient_thresh);
    int nSpec = cd.numSpecies();
    measured_comps[0] = -1;
    for (int i=0; i<nSpec && measured_comps[0]<0; ++i){
      const std::string& name = cd.speciesNames()[i];
      if (name=="OH") {
          measured_comps[0] = i + sCompY;
      }
    }
    if (measured_comps[0] < 0) {
      BoxLib::Abort("OH needed for diagnostic, but not found in chemical mech");
    }
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "thresh_O") {
    transient_thresh = dpdt_thresh_DEF;
    pp.query("O_thresh",transient_thresh);
    int nSpec = cd.numSpecies();
    for (int i=0; i<nSpec; ++i){
      const std::string& name = cd.speciesNames()[i];
      if (name=="O") {
          measured_comps[0] = i + sCompY;
      }
    }
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "onset_CO2") {
    transient_thresh = dpdt_thresh_DEF;
    pp.query("CO2_thresh",transient_thresh);
    int nSpec = cd.numSpecies();
    for (int i=0; i<nSpec; ++i){
      const std::string& name = cd.speciesNames()[i];
      if (name=="CO2") {
          measured_comps[0] = i + sCompY;
      }
    }
    num_measured_values = measured_comps.size();
  }
  else if (diagnostic_name == "mean_difference") {
    mean_delta_cond_start = 0.0;
    mean_delta_cond_stop = 0.0;
    std::string mean_delta_cond_spec;
    std::string mean_delta_numer_spec;
    std::string mean_delta_denom_spec;
    pp.query("mean_delta_cond_start",mean_delta_cond_start);
    pp.query("mean_delta_cond_stop",mean_delta_cond_stop);
    pp.query("mean_delta_cond_spec",mean_delta_cond_spec);
    pp.query("mean_delta_numer_spec",mean_delta_numer_spec);
    pp.query("mean_delta_denom_spec",mean_delta_denom_spec);

    measured_comps.resize(3,-100);
    int nSpec = cd.numSpecies();
    for (int i=0; i<nSpec; ++i){
      const std::string& name = cd.speciesNames()[i];
      if (name==mean_delta_cond_spec) {
          measured_comps[0] = i + sCompY;
      }
      if (name==mean_delta_numer_spec) {
          measured_comps[1] = i + sCompY;
      }
      if (name==mean_delta_denom_spec) {
          measured_comps[2] = i + sCompY;
      }
    }

    if( measured_comps[0] >= 0 ){
        IntVect iv(D_DECL(0,0,0));
        Real X_cond_init;
        pp.get(mean_delta_cond_spec.c_str(), X_cond_init);
        // Rework cond start/stop to be mole fraction instead
        // of fractional conversion
        //std::cout << " Looking for c based on X " << X_cond_init << std::endl;
        mean_delta_cond_start = (1.0 - mean_delta_cond_start )*X_cond_init;
        mean_delta_cond_stop = (1.0 - mean_delta_cond_stop )*X_cond_init;
    }

    if (mean_delta_cond_spec == "time") {
        measured_comps[0] = -2; // -1 was pressure, this should be an enum
    }
    if (mean_delta_numer_spec == "time") {
        measured_comps[1] = -2; // -1 was pressure, this should be an enum
    }
    if (mean_delta_denom_spec == "time") {
        measured_comps[2] = -2; // -1 was pressure, this should be an enum
    }
    if (mean_delta_cond_spec == "unity") {
        measured_comps[0] = -3; // -1 was pressure, this should be an enum
    }
    if (mean_delta_numer_spec == "unity") {
        measured_comps[1] = -3; // -1 was pressure, this should be an enum
    }
    if (mean_delta_denom_spec == "unity") {
        measured_comps[2] = -3; // -1 was pressure, this should be an enum
    }
    num_measured_values = 1;
  }
  else if (diagnostic_name == "record_solution") {
      measured_comps[0] = sCompT;
      num_measured_values = measured_comps.size();
  }
  else {
    int comp = cd.index(diagnostic_name);
    if (comp < 0) {
      std::string err = "Invalid species/temp for: " + pp_prefix;
      BoxLib::Abort(err.c_str());
    }
    else {
      measured_comps[0] = sCompY+comp;
      num_measured_values = measurement_times.size() * measured_comps.size();
    }
  }

  measurement_error = ZeroDReactorErr_DEF;
  pp.query("measurement_error",measurement_error);

  if (pp.countval("log_file")>0) {
    pp.get("log_file",log_file);
  }
  if (pp.countval("solution_savefile")>0) {
    pp.get("solution_savefile",solution_savefile);
    save_this = true;
    std::cout << "Saving solution to file " << solution_savefile.c_str() << std::endl;
  }
  else {
      save_this = false;
      //std::cout << "Not saving solution to file " << solution_savefile.c_str() << std::endl;
  }

}



void
ZeroDReactor::GetMeasurementError(std::vector<Real>& observation_error)
{
  for (int i=0; i<NumMeasuredValues(); ++i) {
    observation_error[i] = measurement_error;
  }
}

bool
ZeroDReactor::ValidMeasurement(Real data) const
{
  // A reasonable test whether result is p, T, X or time
  return ( data > 0 && data < 1.e5 );
}

std::pair<bool,int>
ZeroDReactor::GetMeasurements(std::vector<Real>& simulated_observations,int data_num_points, Real data_tstart, Real data_tend)
{ 
   

  BL_ASSERT(is_initialized);
  Reset();
  const Box& box = funcCnt.box();
  int Nspec = cd.numSpecies();
  //std::cout << "\n\n Running ZeroDReactor "  << diagnostic_name << std::endl;

  bool inside_range_leen = false;
  bool outside_range_leen = false;
	
  measurement_times.resize(data_num_points);
  Real dt = data_tend - data_tstart;  BL_ASSERT(dt>=0);
  for (int i=0; i<data_num_points; ++i) {
    measurement_times[i] = data_tstart + i*dt/(data_num_points-1);
  }

  //std::cout << "numpts new = " << data_num_points << std::endl;
  //std::cout << "tend new = " << data_tend << std::endl;


  bool leen_test = diagnostic_name != "pressure_rise"
    && diagnostic_name != "max_pressure" 
    && diagnostic_name != "max_OH" 
    && diagnostic_name != "thresh_O" 
    && diagnostic_name != "inflect_OH" 
    && diagnostic_name != "onset_OH" 
    && diagnostic_name != "onset_CO2" 
    && diagnostic_name != "onset_pressure_rise"
    && diagnostic_name != "mean_difference"
	&& diagnostic_name != "record_solution";


  if (diagnostic_name == "temp") {
    num_measured_values = measurement_times.size() * measured_comps.size();
  }
  else if (diagnostic_name == "pressure") {
    num_measured_values = measurement_times.size() * measured_comps.size();
  }
  else if (leen_test) { 
    int comp = cd.index(diagnostic_name);
    if (comp < 0) {
      std::string err = "Invalid species/temp for: ";
      BoxLib::Abort(err.c_str());
    }
    else {
      num_measured_values = measurement_times.size() * measured_comps.size();
    }
  }


  int num_time_nodes = measurement_times.size();
  simulated_observations.resize(NumMeasuredValues());
 
  bool sample_evolution = diagnostic_name != "pressure_rise"
    && diagnostic_name != "max_pressure" 
    && diagnostic_name != "max_OH" 
    && diagnostic_name != "thresh_O" 
    && diagnostic_name != "inflect_OH" 
    && diagnostic_name != "onset_OH" 
    && diagnostic_name != "onset_CO2" 
    && diagnostic_name != "onset_pressure_rise"
    && diagnostic_name != "mean_difference";

  std::ofstream ofs;
  std::ofstream sfs;
  bool log_this = (log_file != log_file_DEF);
  if (log_this) {
    EnsureFolderExists(log_file);
    ofs.open(log_file.c_str());
  }
  if (save_this) {
    EnsureFolderExists(solution_savefile);
    sfs.open(solution_savefile.c_str());
    std::cout << "Opened file to save solution to file " << solution_savefile.c_str() << std::endl;
    sfs << "i time T";
    for (int is=0; is<Nspec; ++is){
        sfs << " " << cd.speciesNames()[is].c_str();
    }
    sfs << std::endl;
  }
  bool finished;
  int finished_count; // Used only for onset_CO2 when added

  if (verbosity > 2 && ParallelDescriptor::IOProcessor()) {
    std::string filename = diagnostic_prefix + name + ".dat";
    //std::cout << "Writing solution for " << name << " to " << filename << std::endl;
    EnsureFolderExists(filename);
    std::ofstream osf; osf.open(filename.c_str());
    osf.close();
  }

  if (reactor_type == CONSTANT_VOLUME) {
    FArrayBox& rYold = s_init;
    FArrayBox& rYnew = s_final;
    FArrayBox& rHold = s_init;
    FArrayBox& rHnew = s_final;
    FArrayBox& Told  = s_init;
    FArrayBox& Tnew  = s_final;
    FArrayBox* diag = 0;

    s_init.copy(s_save);
    s_final.copy(s_save);
    Real t_end = 0;
    int i = 0;
    if (t_end == measurement_times[i] && sample_evolution) {
      simulated_observations[i] = ExtractMeasurement();
      if (! ValidMeasurement(simulated_observations[i])) {
        return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_1"));
      }
      i++;
    }

    Real val_new, val_old, dval_old, val_old2, t_startlast, ddval_old, dt_old, dval, ddval;
    Real max_curv;
    if (diagnostic_name == "pressure_rise" 
            || diagnostic_name == "onset_pressure_rise"
            || diagnostic_name == "max_pressure"  
            || diagnostic_name == "inflect_OH"  
            || diagnostic_name == "thresh_O"  
            || diagnostic_name == "onset_OH"  
            || diagnostic_name == "onset_CO2"  
            || diagnostic_name == "max_OH" ) {
      val_new = ExtractMeasurement();
      dval_old = 0;
      ddval_old = 0;
      max_curv = 0;
      i++;
    }
    Real dt = 0;

    finished = false;
    finished_count = 0;
    t_startlast = 0.;
    bool first = true;
    for ( ; i<num_time_nodes && !finished; ++i) {

      if (num_time_nodes != 1  &&  i == num_time_nodes - 1) {
	return std::pair<bool,int>(false,ErrorID("REACTOR_DID_NOT_COMPLETE"));
      }

      Real t_start = t_end;
      t_end = measurement_times[i];
      dt_old = dt;
      dt = t_end - t_start;   
      if( dt_old == 0 ) dt_old = dt;

      bool ok = cd.solveTransient_sdc(rYnew,rHnew,Tnew,rYold,rHold,Told,C_0,
				      funcCnt,box,sCompY,sCompRH,sCompT,
				      dt,Patm,diag,true);

      if (!ok) {
	return std::pair<bool,int>(false,ErrorID("VODE_FAILED"));
      }

      if (sample_evolution) {
        simulated_observations[i] = ExtractMeasurement();
        if (! ValidMeasurement(simulated_observations[i])) {
          return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_2"));
        }
      }
      if (save_this) {
          std::vector<Real> thesol;
          ExtractXTSolution(thesol);
          int nSpec = cd.numSpecies();
          sfs << i << " " << 0.5*(t_start+t_end) << " " << thesol[nSpec];
          for (int is=0; is<nSpec; ++is){
              sfs << " " << thesol[is];
          }
          sfs << std::endl;
      }

      if (first) {
        first = false;
        val_old = val_new;
      }
      val_old2 = val_old;
      val_old = val_new;
      val_new = ExtractMeasurement();
      dval_old = dval;
      ddval_old = ddval;
      dval = (val_new -  val_old2) / (dt+dt_old);
      ddval = (val_new - 2.0*val_old + val_old2) / (dt_old*dt);
      if (log_this) {
        ofs << i << " " << 0.5*(t_start+t_end) << " " << dval << "  "
            << ddval << " " << val_old << " " << val_new << std::endl;
      }

      if (diagnostic_name == "onset_pressure_rise"
          || diagnostic_name == "pressure_rise"
          || diagnostic_name == "onset_OH"   ) {

          finished = dval > transient_thresh && dval < dval_old;
      }
      if( diagnostic_name == "onset_CO2") {
          if (val_new > transient_thresh && ddval - ddval_old < 0) {
              // May be finished, but there is some occasional spurious drops that don't
              // reflect a true maximum - count this occurance
              finished_count++;
          }
          else {
              finished_count=0; // reset finished_count if we go up
          }

          if( finished_count > 5 ){
              finished = true;
          }
      }
      else if (diagnostic_name == "max_pressure") {

          finished = val_old > transient_thresh && (val_new - val_old)/dt < transient_thresh;
      }
      else if (diagnostic_name == "max_OH") {

          finished = val_old > transient_thresh && (val_new - val_old)/dt < 0;
      }
      else if (diagnostic_name == "inflect_OH") {

          if( ddval > max_curv ) {
              max_curv = ddval;
          }
          finished = max_curv > transient_thresh && ddval < 0.05*max_curv; // max_curv*0.001;
      }
      else if (diagnostic_name == "thresh_O") {

          finished = val_new > transient_thresh;
      }
      
      if (finished) {
        simulated_observations[0] = t_startlast;
        if (! ValidMeasurement(simulated_observations[0])) {
          return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_3"));
        }
        simulated_observations[0] *= 1.e6;
      }

      dval_old = dval;
      ddval_old = ddval;
      
      rYold.copy(rYnew,sCompY,sCompY,Nspec);
      rHold.copy(rHnew,sCompRH,sCompRH,1);
      Told.copy(Tnew,sCompT,sCompT,1);

      t_startlast = t_start;
    }
  }
  // This is constant volume / constant pressure conditional
  else {
    BL_ASSERT(reactor_type == CONSTANT_PRESSURE);
    FArrayBox& Yold = s_init;
    FArrayBox& Ynew = s_final;
    FArrayBox& Told = s_init;
    FArrayBox& Tnew = s_final;

    s_init.copy(s_save);
    s_final.copy(s_save);
    Real t_end = 0;
    int i = 0;
    if (t_end == measurement_times[i] && sample_evolution) {
      simulated_observations[i] = ExtractMeasurement();
      if (! ValidMeasurement(simulated_observations[i])) {
        return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_4"));
      }
      i++;
    }

    std::vector<Real> mean_difference_sol_old;
    std::vector<Real> mean_difference_sol;
    bool inside_range;
    Real numer_start, denom_start;
    Real numer_stop, denom_stop;
    Real mean_difference_denom, mean_difference_numer;
    if (diagnostic_name == "mean_difference") {
        mean_difference_sol.resize(measured_comps.size());
        ExtractMeasurements(mean_difference_sol, 0);
        mean_difference_sol_old.resize( mean_difference_sol.size());
        std::fill( mean_difference_sol_old.begin(), mean_difference_sol_old.end(), 0.0);
        inside_range = false;
        numer_start = -1;
        numer_stop = -1;
        i++;
    }
    
    finished = false;
    bool first = true;
    for ( ; i<num_time_nodes; ++i) {
      Real t_start = t_end;
      t_end = measurement_times[i];
      Real dt = t_end - t_start;
      bool ok = cd.solveTransient(Ynew,Tnew,Yold,Told,funcCnt,box,
				  sCompY,sCompT,dt,Patm);		
	
      if (!ok) {
	return std::pair<bool,int>(false,ErrorID("VODE_FAILED"));
      }

      if (sample_evolution) {
        simulated_observations[i] = ExtractMeasurement();
        if (! ValidMeasurement(simulated_observations[i])) {
          return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_5"));
        }
      }

      if (save_this) {
          std::vector<Real> thesol;
          ExtractXTSolution(thesol);
          int nSpec = cd.numSpecies();
          sfs << i << " " << 0.5*(t_start+t_end) << " " << thesol[nSpec];
          for (int is=0; is<nSpec; ++is){
              sfs << " " << thesol[is];
          }
          sfs << std::endl;
      }

      if (log_this) {
          Real cond = ExtractMeasurement();
        ofs << i << " " << 0.5*(t_start+t_end) << " " << cond << std::endl;
      }
      
      // Diagnostics here
      if (diagnostic_name == "mean_difference") {
          if (first) {
#if 0
              std::cout << "Using mean difference diagnostic conditional on " 
                  << measured_comps[0]  << " between " << mean_delta_cond_start << " , "
                  << mean_delta_cond_stop << " numerator: " << measured_comps[1] << 
                  " denominator: "<< measured_comps[2] << std::endl;
#endif              
              first = false;
          }

          // Get solution 
          mean_difference_sol_old = mean_difference_sol;
          ExtractMeasurements(mean_difference_sol, t_end);
		 
          const int cond_id = 0;
          const int numer_id = 1;
          const int denom_id = 2;

          // Check if we have gone past the start of the condition
          if( mean_difference_sol[cond_id] < mean_delta_cond_start 
                  && mean_difference_sol[cond_id] > mean_delta_cond_stop
                  && !inside_range ) {
              numer_start = 
                  (mean_difference_sol[numer_id]
                   - mean_difference_sol_old[numer_id]) / 
                  ( mean_difference_sol[cond_id] 
                    - mean_difference_sol_old[cond_id]) *
                  ( mean_delta_cond_start 
                    - mean_difference_sol_old[cond_id] )
                  + mean_difference_sol_old[numer_id];

              denom_start = 
                  (mean_difference_sol[denom_id]
                   - mean_difference_sol_old[denom_id]) / 
                  ( mean_difference_sol[cond_id] 
                    - mean_difference_sol_old[cond_id]) *
                  ( mean_delta_cond_start 
                    - mean_difference_sol_old[cond_id] )
                  + mean_difference_sol_old[denom_id];

              //std::cout << "Start of range at " 
              //    << numer_start << "/" << denom_start <<  " inside: " <<  inside_range << std::endl;

              inside_range = true;
			  inside_range_leen = true;
	          
          }
		
	      if( (mean_difference_sol[cond_id] < mean_delta_cond_stop) ) {
            outside_range_leen = true; }

          if( (mean_difference_sol[cond_id] < mean_delta_cond_stop)
                  && inside_range ) {				
              BL_ASSERT( numer_start > 0 );
              BL_ASSERT( denom_start > 0 );
              inside_range = false;
			

              numer_stop = 
                  (mean_difference_sol[numer_id]
                   - mean_difference_sol_old[numer_id]) / 
                  ( mean_difference_sol[cond_id] 
                    - mean_difference_sol_old[cond_id]) *
                  ( mean_delta_cond_stop 
                    - mean_difference_sol_old[cond_id] )
                  + mean_difference_sol_old[numer_id];

              denom_stop = 
                  (mean_difference_sol[denom_id]
                   - mean_difference_sol_old[denom_id]) / 
                  ( mean_difference_sol[cond_id] 
                    - mean_difference_sol_old[cond_id]) *
                  ( mean_delta_cond_stop 
                    - mean_difference_sol_old[cond_id] )
                  + mean_difference_sol_old[denom_id];

              //std::cout << "Stop of range at " 
              //    << numer_stop << "/" << denom_stop << std::endl;
              // Denomenator difference (unity if -3 - get this in a enum)
              if( measured_comps[denom_id] != -3 ){
                  mean_difference_denom = denom_stop - denom_start;
              }
              else {
                  mean_difference_denom = 1.0;
              }
              mean_difference_numer = fabs(numer_stop - numer_start);

              if( measured_comps[denom_id] == -2 ) {
                  mean_difference_denom *= 1.0e3; // convert to ms
              }
              else if( measured_comps[denom_id] > 0 ) {
                  mean_difference_denom *= 1.0e6; // convert X to ppm
              }

              if( measured_comps[numer_id] == -2 ) {
                  mean_difference_numer *= 1.0e3; // convert to ms
              }
              else if( measured_comps[numer_id] > 0 ) {
                  mean_difference_numer *= 1.0e6; // convert X to ppm
              }

              //std::cout << "Computed measurement: " <<  simulated_observations[0] <<
              //    " using : " << mean_difference_numer << "/" << mean_difference_denom << std::endl;
              finished = true;
              if( fabs(mean_difference_denom) > 0 ){
                  simulated_observations[0] = mean_difference_numer 
                      / mean_difference_denom;
              }
              inside_range = false;
			  outside_range_leen = true;
          }

          if( finished ){
              if (! ValidMeasurement(simulated_observations[0]) 
                      || fabs(mean_difference_denom) < 1.0e-20) {
                return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_6"));
              }
          }
      }


      // Done diagnostics - carry on
      // std::cout << i << " Computed measurement: " <<  simulated_observations[0]  << " finished: " << finished << std::endl;
      
      Yold.copy(Ynew,sCompY,sCompY,Nspec);
      Told.copy(Tnew,sCompT,sCompT,1);
    } // End of loop over time samples extracted
  }

  if (log_this) {
    ofs.close();
  }
  if (save_this) {
    sfs.close();
  }

  //std::cout << "--> End Computed measurement: " <<  simulated_observations[0]  << " finished: " << finished << std::endl;
  if (diagnostic_name == "mean_difference" && !finished) {
    if (!inside_range_leen && outside_range_leen){
      return std::pair<bool,int>(false,ErrorID("NEEDED_MEAN_REFINE"));
    }
    else if (inside_range_leen || !outside_range_leen){
    return std::pair<bool,int>(false,ErrorID("NEEDED_MEAN_BUT_NOT_FINISHED"));
    }
  }
  return std::pair<bool,int>(true,ErrorID("SUCCESS"));
}

void
ZeroDReactor::ComputeMassFraction(FArrayBox& Y) const
{
  int Nspec = cd.numSpecies();
  Box box = s_final.box();
  Y.resize(box,Nspec);
  if (reactor_type == CONSTANT_VOLUME) { // In this case, state holds rho.Y
    for (IntVect iv=box.smallEnd(), End=box.bigEnd(); iv<=End; box.next(iv)) {
      Real rho = 0;
      for (int i=0; i<Nspec; ++i) {
        rho += s_final(iv,sCompY+i);
      }
      for (int i=0; i<Nspec; ++i) {
        Y.copy(s_final,sCompY+i,i,1);
        Y.mult(1/rho,i,1);
      }
    }
  }
  else { // In this case, state holds Y
    Y.copy(s_final,box,sCompY,box,0,Nspec);
  }
}

void
ZeroDReactor::ExtractXTSolution(std::vector<Real>& sol) 
{
  BL_ASSERT(is_initialized);

  int Nspec = cd.numSpecies();
  sol.resize(Nspec+1);
  sol[Nspec] = s_final(s_final.box().smallEnd(),sCompT);
 

  FArrayBox Y;
  ComputeMassFraction(Y);
  const Box& box = Y.box();


  // Compute mole fraction
  FArrayBox X(box,Nspec);
  cd.massFracToMoleFrac(X,Y,box,0,0);


  int iSpec = measured_comps[0] - sCompY;
  for( int i = 0; i<Nspec; ++i){
      sol[i] = X(box.smallEnd(), i);
  }
  return;

  // Return molar concentration
  // FArrayBox C(box,Nspec);
  // cd.massFracToMolarConc(C,Y,s_final,density,box,0,0,sCompT,0);
  // return C(box.smallEnd(),iSpec);
}

Real
ZeroDReactor::ExtractMeasurement() const
{
  BL_ASSERT(is_initialized);

  if (measured_comps[0] == sCompT) { // Return temperature
    return s_final(s_final.box().smallEnd(),measured_comps[0]);
  }
  else if ((measured_comps[0] < 0) && (reactor_type == CONSTANT_PRESSURE)) {
    return Patm;
  }

  FArrayBox Y;
  ComputeMassFraction(Y);
  const Box& box = Y.box();
  int Nspec = cd.numSpecies();

  // Compute mole fraction
  FArrayBox X(box,Nspec);
  cd.massFracToMoleFrac(X,Y,box,0,0);

  int iSpec = measured_comps[0] - sCompY;
  //if (measured_comps[0] > 0 && diagnostic_name != "max_OH") {
  //if (measured_comps[0] > 0) {
  //  return X(box.smallEnd(),iSpec);
  //}

  // Get pressure and density:
  //     CV: s_final contains rho.Y, P = P(rho,T,Y)
  //     CP: P=Patm, rho = rho(P,T,Y)
  FArrayBox density(box,1);
  FArrayBox pressure(box,1);
  if (reactor_type == CONSTANT_VOLUME) {
    density.setVal(0);
    for (IntVect iv=box.smallEnd(), End=box.bigEnd(); iv<=End; box.next(iv)) {
      for (int i=0; i<Nspec; ++i) {
        density(iv,0) += s_final(iv,sCompY+i);
      }
    }      
    cd.getPGivenRTY(pressure,density,s_final,Y,box,0,sCompT,0,0);
  } else {
    cd.getRhoGivenPTY(density,Patm,s_final,Y,box,sCompT,0,0);
    pressure.setVal(Patm * 101325,0);
  }

  if (verbosity > 2)
  {
    std::string filename = diagnostic_prefix + name + ".dat";
    std::ofstream ofs; ofs.open(filename.c_str(),std::ios::app);
    IntVect se = box.smallEnd();
    for (int i=box.smallEnd()[0]; i<=box.bigEnd()[0]; ++i) {
      IntVect iv(se); iv[0] = i;
      ofs << density(iv,0) << " " << s_final(iv,sCompT) << " ";
      for (int n=0; n<Nspec; ++n) {
	ofs << Y(iv,n) << " ";
      }
      ofs << pressure(iv,0) << std::endl;
    }
    ofs.close();
  }

  if (measured_comps[0] < 0) { // Return pressure
    return pressure(box.smallEnd(),0) / 101325;
  }

  // Return molar concentration
  FArrayBox C(box,Nspec);
  cd.massFracToMolarConc(C,Y,s_final,density,box,0,0,sCompT,0);
  return C(box.smallEnd(),iSpec);
}

void
ZeroDReactor::Reset()
{
  if (is_initialized) {
    funcCnt.setVal(0);
  }
}

void
ZeroDReactor::InitializeExperiment()
{
  const int nSpec = cd.numSpecies();
  const int nComp = nSpec + 4;

  if (Tfile > 0) {
    std::ifstream is;
    is.open(pmf_file_name.c_str());
    FArrayBox fileFAB;
    fileFAB.readFrom(is);
    is.close();

    // Simple check to see if number of species is same between compiled mech and fab file
    if (nComp != fileFAB.nComp()) {
      std::cout << "pmf file is not compatible with the mechanism compiled into this code" << '\n';
      std::cout << "pmf file number of species: " << fileFAB.nComp() - 4 << '\n';
      std::cout << "expecting: " << nSpec << '\n';
      BoxLib::Abort();
    }

    // Find location
    bool found = false;
    const Box& boxF = fileFAB.box();
    IntVect iv=boxF.smallEnd();
    for (IntVect End=boxF.bigEnd(); iv<=End && !found; boxF.next(iv)) {
      if (fileFAB(iv,sCompT)>=Tfile) found = true;
    }

    Box box(iv,iv);
    s_init.resize(box,fileFAB.nComp()); s_init.copy(fileFAB);
    s_init.mult(1.e3,sCompR,1); // to mks
    funcCnt.resize(box,1);
  }
  Box bx = s_init.box();
  
  if (reactor_type == CONSTANT_VOLUME) {
    cd.getHmixGivenTY(s_init,s_init,s_init,bx,sCompT,sCompY,sCompRH);
    s_init.mult(s_init,sCompR,sCompRH,1);
    for (int i=0; i<nSpec; ++i) {
      s_init.mult(s_init,sCompR,sCompY+i,1);
    }
    C_0.resize(bx,nSpec+1); C_0.setVal(0);
  }

  s_final.resize(bx,s_init.nComp());
  s_final.copy(s_init);

  s_save.resize(bx,s_init.nComp());
  s_save.copy(s_init);

  is_initialized = true;
}

static
void parseLMC(const std::vector<std::string>& tokens, Array<Real>& lmc_data, int nY)
{
  BL_ASSERT(tokens.size() == 2*nY + 8);
  for (int i=0; i<nY; ++i) {
    lmc_data[i] = atof(tokens[1+i].c_str()); // mass fractions
  }
  lmc_data[nY]   = atof(tokens[nY+1].c_str()); // Density
  lmc_data[nY+1] = atof(tokens[nY+2].c_str()); // Temperature
  lmc_data[nY+2] = atof(tokens[nY+4].c_str()); // Velocity
  lmc_data[nY+3] = atof(tokens[0].c_str()); // location
}

bool
PREMIXReactor::ReadBaselineSoln(const std::string& filename)
{
  if (!BoxLib::FileExists(filename)) {
    return false;
  }

  // Read solution
  if (ParallelDescriptor::IOProcessor()) {
    std::cerr << "Reading baseline solution for " << name << " from: " << filename << std::endl;
  }
  return baseline_premix_sol->ReadSoln(filename);
}

PREMIXReactor::PREMIXReactor(ChemDriver& _cd, const std::string& pp_prefix)
  : SimulatedExperiment(), name(pp_prefix), cd(_cd), max_premix_iters(max_premix_iters_DEF)
{
  ParmParse pp(pp_prefix.c_str());
  pp.query("verbosity",verbosity);

  measurement_error = PREMIXReactorErr_DEF;
  pp.query("measurement_error",measurement_error);

  int num_sol_pts = 1000; pp.query("num_sol_pts",num_sol_pts);
  int nComp = cd.numSpecies() + 3;
  premix_sol = new PremixSol(nComp,num_sol_pts);
  baseline_premix_sol = new PremixSol(nComp,num_sol_pts);
  have_baseline_sol = false;
  lrstrtflag=0;

  if (pp.countval("baseline_soln_file")) {
    pp.get("baseline_soln_file",baseline_soln_file);
    have_baseline_sol = ReadBaselineSoln(baseline_soln_file);
  }

  pp.get("premix_input_path",premix_input_path);
  pp.get("premix_input_file",premix_input_file);

  //Check for prerequisites for this experiment
  //    These are sometimes necessary to get a reasonable initial condition
  //    that premix can converge from
  int nprereq = pp.countval("prereqs");
  //std::cerr << "Experiment " <<  pp_prefix  << std::endl;
  Array<std::string> prereq_names;
  if( nprereq > 0 ){
      pp.getarr("prereqs",prereq_names,0,nprereq);
      for( int i = 0; i < nprereq; i++ ){
          std::string prefix = prereq_names[i];
          ParmParse pppr(prefix.c_str() );
          std::string type; pppr.get("type", type );
          if( type == "PREMIXReactor" ){
              PREMIXReactor *prereq_reactor 
                  = new PREMIXReactor(cd,prereq_names[i]);
              prereq_reactors.push_back(prereq_reactor);
          }
          else{
              std::cerr << " PREMIXReactor can not use " << type << " as prereq \n";
          }
          BL_ASSERT( type == "PREMIXReactor" );
      }
      if (ParallelDescriptor::IOProcessor()) {
        std::cerr << "Experiment " <<  pp_prefix  << " registering " << nprereq << " prerequisites " << std::endl;
      }
  }
  else {
    lmc_soln_file = ""; pp.query("lmc_soln",lmc_soln_file);
    if (lmc_soln_file != "") {
      std::ifstream ifs(lmc_soln_file.c_str());
      if (!ifs.is_open()) {
	BoxLib::Abort("error while opening lmc file");
      }
      int lmc_step; ifs >> lmc_step;
      int lmc_npts; ifs >> lmc_npts;
      nmax = premix_sol->maxgp;
      if (lmc_npts > nmax) {
	BoxLib::Abort("lmc file has more points than max allowed");
      }
      Real lmc_time; ifs >> lmc_time;

      int nY = -1;
      std::string line;
      bool ok = std::getline(ifs, line);
      while (ok && line.size() == 0) {ok = std::getline(ifs, line);}

      std::vector<std::string> tokens = BoxLib::Tokenize(line, " ");
      nY = (tokens.size() - 8) / 2;
      BL_ASSERT(2*nY+8 == tokens.size());

      Array<Array<Real> > lmc_data(lmc_npts, Array<Real>(nY+4));
      int j = 0;
      parseLMC(tokens,lmc_data[j++],nY);

      while(std::getline(ifs, line)) {
	BL_ASSERT(j<lmc_npts);
	tokens = BoxLib::Tokenize(line, " ");
	parseLMC(tokens,lmc_data[j++],nY);
      }

      Real Pcgs = atof(tokens[nY+6].c_str());

      if (ifs.bad()) {
	BoxLib::Abort("error while reading lmc file");
      }
      ifs.close();

      double * solvec = premix_sol->solvec;
      premix_sol->ngp = lmc_npts;
      for (int j=0; j<lmc_npts; ++j) {
	solvec[ j + 0 * nmax ] = lmc_data[j][nY+3]; // position
	solvec[ j + 1 * nmax ] = lmc_data[j][nY+1]; // temperature
	for (int n=0; n<nY; ++n) {
	  solvec[ j + (n+2) * nmax ] = lmc_data[j][n]; // mass fractions
	}
	solvec[ j + (nY+2) * nmax ] = lmc_data[j][nY] * lmc_data[j][nY+2]; // flow rate
      }
      solvec[lmc_npts + (nY+2)*nmax    ] = Pcgs;
      solvec[lmc_npts + (nY+2)*nmax + 1] = lmc_data[0][nY] * lmc_data[0][nY+2];

    }
  }
  pp.query("max_premix_iters",max_premix_iters);
}

PREMIXReactor::~PREMIXReactor()
{
  delete premix_sol;
  delete baseline_premix_sol;
}

void
PREMIXReactor::GetMeasurementError(std::vector<Real>& observation_error)
{
  for (int i=0; i<NumMeasuredValues(); ++i) {
    observation_error[i] = measurement_error;
  }
}

bool
PREMIXReactor::ValidMeasurement(Real data) const
{
  // A reasonable test for data = flame speed
  return ( data > 0 && data < 1.e5 );
}

/*
 *
 * Run a premix case with whatever chemistry is set up and store the
 * solution in baseline_premix_sol
 *
 */
void
PREMIXReactor::SaveBaselineSolution(const std::string& prefix)
{
    std::vector<Real> simulated_obs;
    std::pair<bool,int> status;

    if (!have_baseline_sol) {
	  ParmParse ppe(prefix.c_str());
	  Real data_tstart = 0; ppe.query("data_tstart",data_tstart);
  	  Real data_tend = 0; ppe.query("data_tend",data_tend); BL_ASSERT(data_tend>0);
  	  int data_num_points = -1;
  	  ppe.query("data_num_points",data_num_points); BL_ASSERT(data_num_points>0);

      status = GetMeasurements(simulated_obs,data_num_points, data_tstart, data_tend);
      if(status.first){
        if (baseline_soln_file != "") {
	  std::cerr << "Writing baseline solution for " << name << " to: " << baseline_soln_file << std::endl;
	  premix_sol->WriteSoln(baseline_soln_file);
        }
	have_baseline_sol = true;
	solCopyOut(baseline_premix_sol);
      }
      else{
        std::string err = "Baseline calculation failed for input file: " + premix_input_file;
        BoxLib::Abort(err.c_str());
      }
    }
}

std::pair<bool,int>
PREMIXReactor::GetMeasurements(std::vector<Real>& simulated_observations,int data_num_points, Real data_tstart, Real data_tend)
{
  //BL_PROFILE("PREMIXReactor::GetMeasurements()");

  // This set to return a single value - the flame speed
  simulated_observations.resize(1);

  int lregrid;
  int lrstrt = 0;
  int v = Verbosity();

#ifndef PREMIX_RESTART
  /*
   * Something about the restart makes the solution less
   * robust, even if it's faster. Taking this out for now.
   * (It was supposed to try to restart if it had a previously 
   * successful solution for this experiment)
   */
  lrstrtflag = 0; 
#endif
  if(have_baseline_sol)
  {
      solCopyIn(baseline_premix_sol);
      lrstrtflag = 1; 
      //std::cerr << "Have baseline solution, " <<  baseline_premix_sol->ngp <<"/" << (premix_sol->ngp) << " gridpoints\n";
  } else
  {
    std::cerr << "No baseline solution for " << name << std::endl;
// BoxLib::Abort();
  }

  // When doing a fresh start, run through prereqs. First starts fresh, subsequent start from
  // solution from the previous. Once the prereqs are done, set restart flag so that solution
  // will pick up from where  prereqs finished. 
  if( lrstrtflag == 0 )
  {
    if( prereq_reactors.size() > 0 )
    {
      if (v > 0 && ParallelDescriptor::IOProcessor())
      {
	std::cerr << " experiment has " << prereq_reactors.size() << " prereqs " << std::endl;
      }

      for( Array<PREMIXReactor*>::iterator pr=prereq_reactors.begin(); pr!=prereq_reactors.end(); ++pr )
      {
	if( lrstrt == 1  ){
	  (*pr)->solCopyIn(premix_sol);
	  (*pr)->lrstrtflag = 1;
	}
	else {
	  (*pr)->lrstrtflag = 0;
	  lrstrt = 1; // restart on the next time through
	}

	std::vector<Real> pr_obs;
	if (v > 0 && ParallelDescriptor::IOProcessor()) {
	  std::cerr << " Running " << (*pr)->premix_input_file
		    << " with restart = " << (*pr)->lrstrtflag << std::endl;
	}

	std::pair<bool,int> retVal = (*pr)->GetMeasurements(pr_obs, data_num_points, data_tstart, data_tend);
	if (!retVal.first) {
	  return std::pair<bool,int>(false,ErrorID("PREREQ_FAILED"));
	}

	if (v > 0 && ParallelDescriptor::IOProcessor()) {
	  std::cerr << " Obtained intermediate observable " << pr_obs[0] << std::endl;
	}

	(*pr)->solCopyOut(premix_sol);
      }

      // If restarting from a prereq, don't regrid, but otherwise regrid the solution
      lrstrtflag = 1;
    }
    lregrid = -1;
  }
  else{

    if (v == 1 && ParallelDescriptor::IOProcessor()) {
      std::cerr << "Restarting from previous solution... " << std::endl;
    }

    // Regrid when restarting from a previous solution of this experiment
    // Don't regrid, because now using the baseline mechanism as prev sol
    lregrid = -1;
  }

  BL_ASSERT(premix_sol != 0);
  double * savesol = premix_sol->solvec; 
  int * solsz = &(premix_sol->ngp);

  // Regrid to some size less than the restart solution size
  if( lregrid > 0 )
  {
#if 0
    const int min_reasonable_regrid = min_reasonable_regrid_DEF;
    int regrid_sz = *solsz/4;

    // Regrid to larger of regrid_sz estimate from previous
    // solution or some reasonable minimum, but don't regrid
    // if that would be bigger than previous solution
    lregrid = std::max(min_reasonable_regrid, regrid_sz); 
#endif
    lregrid = 50;
    if( lregrid > *solsz ) lregrid = -1;

    if( lregrid > 0) {
      if (ParallelDescriptor::IOProcessor()) {
	std::cout << "----- Setting up premix to regrid to " 
		  << lregrid <<  " from " <<  *solsz  << std::endl;
      }
    }
  }

  BL_ASSERT(savesol != NULL );
  BL_ASSERT(solsz != NULL );

  if (lmc_soln_file != "") {
    lrstrtflag = 1; 
  }

  int charlen = premix_input_file.size();
  int infilecoded[charlen];
  for(int i=0; i<charlen; i++){
    infilecoded[i] = premix_input_file[i];
  }

  int pathcharlen = premix_input_path.size();
  int pathcoded[pathcharlen];
  for(int i=0; i<pathcharlen; i++){
    pathcoded[i] = premix_input_path[i];
  }

  // Build unit numbers, unique to each thread
  int increment = 1;
  int threadid = 0;
#ifdef _OPENMP
  increment = omp_get_num_threads();
  threadid = omp_get_thread_num();
#endif

  lout  = 6     + threadid;
  lin   = lout  + increment;
  lrin  = lin   + increment;
  lrout = lrin  + increment;
  lrcvr = lrout + increment;
  linck = lrcvr + increment;
  linmc = linck + increment;

  // TODO: Remove all unused units
  open_premix_files_( &lin, &lout, &linmc, &lrin,
                      &lrout, &lrcvr, infilecoded,
                      &charlen, pathcoded, &pathcharlen );

  int is_good = 0;
  int num_steps = 0;
  premix_(&nmax, &lin, &lout, &linmc, &lrin, &lrout, &lrcvr,
          &lenlwk, &leniwk, &lenrwk, &lencwk, 
          savesol, solsz, &lrstrtflag, &lregrid, &is_good, &max_premix_iters, &num_steps);
  
  // Extract the measurements
  // TODO: put into an 'ExtractMeasurements' for consistency with ZeroDReactor

  if( is_good > 0 && *solsz > 0 )
  {
    int nComp = cd.numSpecies() + 3;
    simulated_observations[0]  = savesol[*solsz + nmax*(nComp-1)-1+3];
    if (! ValidMeasurement(simulated_observations[0])) {
      return std::pair<bool,int>(false,ErrorID("INVALID_OBSERVATION_7"));
    }
    lrstrtflag = 1;

    if (Verbosity() > 1 && ParallelDescriptor::IOProcessor()) {
      std::string filename = diagnostic_prefix + name + ".dat";
      //std::cout << "Writing solution for " << name << " to " << filename << std::endl;
      EnsureFolderExists(filename);
      std::ofstream ofs; ofs.open(filename.c_str());
      for (int i=0; i<*solsz; ++i) {
	ofs << savesol[i + nmax*0]; // X
	ofs << " " << savesol[i + nmax*1]; // T
	for (int n=0; n<cd.numSpecies(); ++n) {
	  ofs << " " << savesol[i + nmax*(2+n)]; // Y
	}
	ofs << '\n';
	// p not currently written, as it would break the matrix format
      }
      ofs.close();
    }

  }
  else {
    simulated_observations[0]  = -1;
    lrstrtflag = 0;
    close_premix_files_( &lin, &linck, &lrin, &lrout, &lrcvr );
    if (num_steps == max_premix_iters) {
      return std::pair<bool,int>(false,ErrorID("PREMIX_TOO_MANY_ITERS"));
    }
    return std::pair<bool,int>(false,ErrorID("PREMIX_SOLVER_FAILED"));
  }

  // Cleanup fortran remains
  close_premix_files_( &lin, &linck, &lrin, &lrout, &lrcvr );

  //If this is the first pass, regrid and don't take any steps
#if 0
  if(!have_baseline_sol){
      open_premix_files_( &lin, &lout, &linmc, &lrin,
              &lrout, &lrcvr, infilecoded,
              &charlen, pathcoded, &pathcharlen );

      int is_good = 0;
      int num_steps = 0;
      int premix_iters = 1;
      lrstrtflag = 1; 
      lregrid = 50;
      premix_(&nmax, &lin, &lout, &linmc, &lrin, &lrout, &lrcvr,
              &lenlwk, &leniwk, &lenrwk, &lencwk, 
              savesol, solsz, &lrstrtflag, &lregrid, &is_good, &premix_iters, &num_steps);
      std::cerr << "After regrid pass, solsz = " << *solsz << std::endl;
      // Cleanup fortran remains
      close_premix_files_( &lin, &linck, &lrin, &lrout, &lrcvr );
  
  }
#endif

  return std::pair<bool,int>(true,ErrorID("SUCCESS"));
}

/*
 * CopyData
 * this is to copy the state of the experiment necessary for
 * restart (or anything not present after InitializeExperiment call )
 * so that experiment can be moved
 */
void
PREMIXReactor::CopyData(int src, int dest, int tag)
{
  // things to copy:
  // 1. Solution vector
  // 2. Number of gridpoints

  if (ParallelDescriptor::MyProc() == src) {
    ParallelDescriptor::Send(&(premix_sol->maxgp), 1, dest, tag);
    ParallelDescriptor::Send(premix_sol->solvec, premix_sol->maxgp, 
                              dest, tag);
  }
  else if (ParallelDescriptor::MyProc() == dest) {
    ParallelDescriptor::Recv(&(premix_sol->maxgp), 1, src, tag );
    ParallelDescriptor::Recv((premix_sol->solvec), premix_sol->maxgp, 
                              src, tag);
  }
}


void
PREMIXReactor::InitializeExperiment()
{
    // Pass this as maximum number of gridpoints
    nmax=premix_sol->maxgp;

    // Sizes for work arrays
    lenlwk=4270;
    leniwk=241933;
    lenrwk=90460799;
    lencwk=202;
    lensym=16;
    
    // Check input file
    if( premix_input_file.empty() ){
        std::cerr << "No input file specified for premixed reactor \n";
    }

    int i=0;
    // Initialize all prerequisite simulations also
    for( Array<PREMIXReactor*>::iterator pr=prereq_reactors.begin(); pr!=prereq_reactors.end(); ++pr ){                                                                                
        i++;
        (*pr)->InitializeExperiment();
        //std::cerr << "Initialized prereq " << i << " sz: " << (*pr)->nmax << std::endl;
    }


}

const PremixSol&
PREMIXReactor::getPremixSol() const
{
  return *premix_sol;
}

void 
PREMIXReactor::solCopyIn( PremixSol * solIn ){
    *premix_sol = *solIn;

}

void 
PREMIXReactor::solCopyOut( PremixSol *  solOut){
    *solOut = *premix_sol;
}

void
ZeroDReactor::ExtractMeasurements( std::vector<Real>& measurements, Real sample_time ) const
{
    BL_ASSERT(is_initialized);

    for (int i=0; i<measured_comps.size(); ++i ) {
        if (measured_comps[i] == sCompT) { // Return temperature
            measurements[i] =  s_final(s_final.box().smallEnd(),measured_comps[i]);
        }
        else if ((measured_comps[i] == -1) && (reactor_type == CONSTANT_PRESSURE)) {
            measurements[i] =  Patm;
        }
        else if ((measured_comps[i] == -2) && (reactor_type == CONSTANT_PRESSURE)) {
            measurements[i] =  sample_time;
        }
        else if ((measured_comps[i] == -3) && (reactor_type == CONSTANT_PRESSURE)) {
            measurements[i] =  1.0;
        }
        else {

            FArrayBox Y;
            ComputeMassFraction(Y);
            const Box& box = Y.box();
            int Nspec = cd.numSpecies();

            if (measured_comps[i] < 0 && (reactor_type == CONSTANT_VOLUME)) { // Return pressure
                // CONSTANT_VOLUME case, state holds rho.Y
                FArrayBox rhop(box,2);
                rhop.setVal(0,0);
                for (IntVect iv=box.smallEnd(), End=box.bigEnd(); iv<=End; box.next(iv)) {
                    for (int i=0; i<Nspec; ++i) {
                        rhop(iv,0) += s_final(iv,sCompY+i);
                    }
                }      
                cd.getPGivenRTY(rhop,rhop,s_final,Y,box,0,sCompT,0,1);
                measurements[i] = rhop(box.smallEnd(),1) / 101325;
            }

            // Compute mole fraction
            FArrayBox X(box,Nspec);
            cd.massFracToMoleFrac(X,Y,box,0,0);
            measurements[i] =  X(box.smallEnd(),measured_comps[i] - sCompY);
        }
        // std::cout << " Extracting for component " << i << " id " << measured_comps[i] 
        //     << " value: " << measurements[i] << std::endl;
    }
}
