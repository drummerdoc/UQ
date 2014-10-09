#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <SimulatedExperiment.H>
#include <ParmParse.H>

#include <sys/time.h>

#include <ParallelDescriptor.H>

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


SimulatedExperiment::SimulatedExperiment()
  :  is_initialized(false), log_file(log_file_DEF) {}

SimulatedExperiment::~SimulatedExperiment() {}

void SimulatedExperiment::CopyData(int src, int dest, int tag){}

int
ZeroDReactor::NumMeasuredValues() const {return num_measured_values;}

ZeroDReactor::~ZeroDReactor() {}

const std::vector<Real>&
ZeroDReactor::GetMeasurementTimes() const
{
  return measurement_times;
}

ZeroDReactor::ZeroDReactor(ChemDriver& _cd, const std::string& pp_prefix, const REACTOR_TYPE& _type)
  : SimulatedExperiment(), cd(_cd), reactor_type(_type),num_measured_values(0),
    sCompY(-1), sCompT(-1), sCompR(-1), sCompRH(-1)
{
  ParmParse pp(pp_prefix.c_str());

  std::string expt_type; pp.get("type",expt_type);
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

    measured_comps.resize(3);
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
    if( measured_comps[0] > 0 ){
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

bool
ZeroDReactor::GetMeasurements(std::vector<Real>& simulated_observations)
{
  BL_ASSERT(is_initialized);
  Reset();
  const Box& box = funcCnt.box();
  int Nspec = cd.numSpecies();

  //std::cout << "\n\n Running ZeroDReactor "  << diagnostic_name << std::endl;
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
  bool log_this = (log_file != log_file_DEF);
  if (log_this) {
    ofs.open(log_file.c_str());
  }
  bool finished;
  int finished_count; // Used only for onset_CO2 when added

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
        return false;
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
      i++;
    }
    Real dt = 0;

    finished = false;
    finished_count = 0;
    t_startlast = 0.;
    bool first = true;
    for ( ; i<num_time_nodes && !finished; ++i) {
      Real t_start = t_end;
      t_end = measurement_times[i];
      dt_old = dt;
      dt = t_end - t_start;
      if( dt_old == 0 ) dt_old = dt;
      
      cd.solveTransient_sdc(rYnew,rHnew,Tnew,rYold,rHold,Told,C_0,
                            funcCnt,box,sCompY,sCompRH,sCompT,
                            dt,Patm,diag,true);

      if (sample_evolution) {
        simulated_observations[i] = ExtractMeasurement();
        if (! ValidMeasurement(simulated_observations[i])) {
          return false;
        }
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
          return false;
        }
        simulated_observations[0] *= 1.e6;
      }

      dval_old = dval;
      ddval_old = ddval;
      
      rYold.copy(rYnew,sCompY,sCompY,Nspec);
      rHold.copy(rHnew,sCompRH,sCompRH,Nspec);
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
        return false;
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
    Real dt = 0;



    finished = false;
    bool first = true;
    for ( ; i<num_time_nodes; ++i) {
      Real t_start = t_end;
      t_end = measurement_times[i];
      Real dt = t_end - t_start;

      cd.solveTransient(Ynew,Tnew,Yold,Told,funcCnt,box,
                        sCompY,sCompT,dt,Patm);
      if (sample_evolution) {
        simulated_observations[i] = ExtractMeasurement();
        if (! ValidMeasurement(simulated_observations[i])) {
          return false;
        }
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

          }

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

              std::cout << "Computed measurement: " <<  simulated_observations[0] <<
                  " using : " << mean_difference_numer << "/" << mean_difference_denom << std::endl;
              finished = true;
              if( fabs(mean_difference_denom) > 0 ){
                  simulated_observations[0] = mean_difference_numer 
                      / mean_difference_denom;
              }
              inside_range = false;
          }

          if( finished ){
              if (! ValidMeasurement(simulated_observations[0]) 
                      || fabs(mean_difference_denom) < 1.0e-20) {
                  return false;
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

  //std::cout << "--> End Computed measurement: " <<  simulated_observations[0]  << " finished: " << finished << std::endl;
  if (diagnostic_name == "mean_difference" && !finished) {
      return false;
  }
  return true;
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


PREMIXReactor::PREMIXReactor(ChemDriver& _cd, const std::string& pp_prefix)
  : SimulatedExperiment(), cd(_cd)
{
  ParmParse pp(pp_prefix.c_str());

  ncomp = cd.numSpecies() + 3;

  measurement_error = PREMIXReactorErr_DEF;
  pp.query("measurement_error",measurement_error);

  int num_sol_pts = 1000; pp.query("num_sol_pts",num_sol_pts);
  premix_sol = new PremixSol(ncomp,num_sol_pts);
  lrstrtflag=0;

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
      std::cerr << "Experiment " <<  pp_prefix  << " registering " << nprereq << " prerequisites " << std::endl;
  }

}

PREMIXReactor::~PREMIXReactor()
{
  delete premix_sol;
  // Clean up the mess of prereq_reactors if there are any
//  if( prereq_reactors.size() > 0 ){
//      for( Array<PREMIXReactor*>::iterator pr=prereq_reactors.end();
//              pr!=prereq_reactors.begin(); --pr ){                                                                                
//          delete *pr;
//          prereq_reactors.erase(pr);
//      }
//  }
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


bool
PREMIXReactor::GetMeasurements(std::vector<Real>& simulated_observations)
{
  BL_PROFILE("PREMIXReactor::GetMeasurements()");

  BL_PROFILE_VAR("PREMIXReactor::GetMeasurements()-NoPREMIX", myname);
  BL_PROFILE_VAR("PREMIXReactor::GetMeasurements()-NoPREMIX-a", myname1);
  // This set to return a single value - the flame speed
  simulated_observations.resize(1);

  int lregrid;
  int lrstrt = 0;

#ifndef PREMIX_RESTART
  /*
   * Something about the restart makes the solution less
   * robust, even if it's faster. Taking this out for now.
   * (It was supposed to try to restart if it had a previously 
   * successful solution for this experiment)
   */
  lrstrtflag = 0; 
#endif
  // When doing a fresh start, 
  // run through prereqs. First starts fresh, subsequent start from
  // solution from the previous.
  // Once the prereqs are done, set restart flag so that solution
  // will pick up from where  prereqs finished. 
  if( lrstrtflag == 0 ){
      //std::cerr << "No restart info... " <<std::endl;
      //std::cout << " makepr: " << makepr << " prereq_reactors.size() " << 
      //    prereq_reactors.size() << std::endl;
      if( prereq_reactors.size() > 0 ){
      //    std::cerr << " experiment has " << prereq_reactors.size() << " prereqs " << std::endl;
          for( Array<PREMIXReactor*>::iterator pr=prereq_reactors.begin(); pr!=prereq_reactors.end(); ++pr ){                                                                                
              if( lrstrt == 1  ){
                  (*pr)->solCopyIn(premix_sol);
                  (*pr)->lrstrtflag = 1;
                  //std::cerr <<  "restart this time" << std::endl;
              }
              else{
                  (*pr)->lrstrtflag = 0;
                  lrstrt = 1; // restart on the next time through
  //                std::cerr <<  "restart next time" << std::endl;
              }
              std::vector<Real> pr_obs;
   //           std::cerr << " Running " << (*pr)->premix_input_file  << " with restart = " << (*pr)->lrstrtflag << std::endl;
              bool ok = (*pr)->GetMeasurements(pr_obs);
              if (!ok) {
                return false;
              }
              //  std::cerr << " Obtained intermediate observable " << pr_obs[0] << std::endl;
              (*pr)->solCopyOut(premix_sol);
          }
          lrstrtflag = 1;
          // If restarting from a prereq, don't regrid, but otherwise
          // regrid the solution
      }
      lregrid = -1;
  }
  else{
      std::cerr << "Restarting from previous solution... " 
          << std::endl;
      // Regrid when restarting from a previous solution of 
      // this experiment
      lregrid = 1;
  }
  BL_ASSERT(premix_sol != 0);
  double * savesol = premix_sol->solvec; 
  int * solsz = &(premix_sol->ngp);

  // Regrid to some size less than the restart solution size
  if( lregrid > 0 ){
      const int min_reasonable_regrid = 24;
      int regrid_sz = *solsz/4;

      // Regrid to larger of regrid_sz estimate from previous
      // solution or some reasonable minimum, but don't regrid
      // if that would be bigger than previous solution
      lregrid = std::max(min_reasonable_regrid, regrid_sz); 
      if( lregrid > *solsz ) lregrid = -1;

      if( lregrid > 0 ) {
          std::cout << "----- Setting up premix to regrid to " 
              << lregrid <<  " from " <<  *solsz  << std::endl;
      }
      else{
//          std::cout << "----- Skipping regrid to " 
//              << lregrid <<  " (maybe because it would be too big) " 
//              << *solsz << std::endl;
      }
  }

  BL_ASSERT(savesol != NULL );
  BL_ASSERT(solsz != NULL );

  //std::cerr << "Restart solution size: " << *solsz << std::endl;
  // Pass input dir + file names to fortran
  int charlen = premix_input_file.size();
  int pathcharlen = premix_input_path.size();

  int infilecoded[charlen];
  for(int i=0; i<charlen; i++){
    infilecoded[i] = premix_input_file[i];
  }
  int pathcoded[pathcharlen];
  for(int i=0; i<pathcharlen; i++){
    pathcoded[i] = premix_input_path[i];
  }
  open_premix_files_( &lin, &lout, &linmc, &lrin,
                      &lrout, &lrcvr, infilecoded,
                      &charlen, pathcoded, &pathcharlen );

  BL_PROFILE_VAR_STOP(myname1);
  BL_PROFILE_VAR_STOP(myname);
  // Call the simulation
  //timeval tp;
  //timezone tz;
  //gettimeofday(&tp, NULL);
  //int startPMtime = tp.tv_sec;

  //std::cout << "Calling PREMIX" << std::endl;
  premix_(&nmax, &lin, &lout, &linmc, &lrin, &lrout, &lrcvr,
          &lenlwk, &leniwk, &lenrwk, &lencwk, 
          savesol, solsz, &lrstrtflag, &lregrid);
  //gettimeofday(&tp, NULL);
  //int stopPMtime = tp.tv_sec;
  //std::cout << "PREMIX call took approximately " << (stopPMtime - startPMtime) << " seconds (gettimeofday) " << std::endl;

  //std::cerr << "solsz=" << *solsz << std::endl;
  //// DEBUG Check if something reasonable was saved for solution
  //printf("Grid for saved solution: (%d points)\n", *solsz);
  //FILE * FP = fopen("sol.txt","w");
  //for (int i=0; i<*solsz; i++) {
  //    fprintf(FP,"%d\t", i);
  //    for( int j=0; j<ncomp; j++){
  //        fprintf(FP,"%10.3g\t", savesol[i + j*nmax]);
  //    }
  //    fprintf(FP,"\n");
  //}
  //fclose(FP);
  BL_PROFILE_VAR_START(myname);
  BL_PROFILE_VAR("PREMIXReactor::GetMeasurements()-NoPREMIX-b", myname2);
  
  // Extract the measurements - should probably put into an 'ExtractMeasurements'
  // for consistency with ZeroDReactor
  if( *solsz > 0 ) {
    //std::cout << "Premix generated a viable solution " << std::endl;
    simulated_observations[0]  = savesol[*solsz + nmax*(ncomp-1)-1+3];
    if (! ValidMeasurement(simulated_observations[0])) {
      return false;
    }
    lrstrtflag = 1;
  }
  else{
    //std::cout << "Premix failed to find a viable solution " << std::endl;
    simulated_observations[0]  = -1;
    lrstrtflag = 0;
  }

  // Cleanup fortran remains
  close_premix_files_( &lin, &linck, &lrin, &lrout, &lrcvr );


  // NEXT STEPS:
  // General cleanup
  //     - Take out unused file handles
  //     - Split out ckinit / mcinit calls
  // Try with Davis mechanism
  //     - General code compile with Davis mechanism
  //     - See if I can get a solution
  // Make sure it is robust to changing chemical parameters
  // Put in context of sampling framework
  // Generate 'pseudo-experimental' data
  //      - Need separate object to sample from distribution?
  // Infrastructure to manage set of experiments
  //      - think Marc largely has this done, check that it 
  //        is ok wrt to flame speed measurements
  // Try sampling to get distribution of 1 reaction rate
  //       consistent with observation distribution
  BL_PROFILE_VAR_STOP(myname2);
  BL_PROFILE_VAR_STOP(myname);

  return true;
}

/*
 * CopyData
 * this is to copy the state of the experiment necessary for
 * restart (or anything not present after InitializeExperiment call )
 * so that experiment can be moved
 */
void PREMIXReactor::CopyData(int src, int dest, int tag) {

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
    lenlwk=4055;
    leniwk=241933;
    lenrwk=90460799;
    lencwk=202;
    lensym=16;
    
    // Unit numbers for input/output files
    lin=10;
    lout=6;
    lrin=14;
    lrout=15;
    lrcvr=16;
    linck=25;
    linmc=35;

    // Sizes of data stored in object
    maxsolsz = nmax;
    //ncomp = 12;

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

int
PREMIXReactor::numComp() const
{
  return ncomp;
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
