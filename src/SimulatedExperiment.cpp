#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <SimulatedExperiment.H>
#include <ParmParse.H>

#include <sys/time.h>

static Real Patm_DEF = 1;
static Real dt_DEF   = 0.1;
static Real Tfile_DEF = 900;
static int  num_time_intervals_DEF = 10;
static std::string temp_or_species_name_DEF = "temp";
static Real CVReactorErr_DEF = 15;
static Real PREMIXReactorErr_DEF = 10;

CVReactor::CVReactor(ChemDriver& _cd, const std::string& pp_prefix)
  : SimulatedExperiment(), cd(_cd)
{
  ParmParse pp(pp_prefix.c_str());

  std::string expt_type; pp.get("type",expt_type);
  if (expt_type != "CVReactor") {
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
    measurement_times[i] = data_tstart + (i+1)*dt/data_num_points;
  }

  // Ordering of variables in pmf file used for initial conditions
  sCompT  = 1;
  sCompRH = 2;
  sCompR  = 3;
  sCompY  = 4;

  pp.get("pmf_file_name",pmf_file_name);

  measured_comps.resize(1);
  std::string temp_or_species_name = temp_or_species_name_DEF;
  pp.query("temp_or_species_name",temp_or_species_name);
  if (temp_or_species_name == "temp") {
    measured_comps[0] = sCompT;
  }
  else {
    int comp = cd.index(temp_or_species_name);
    if (comp < 0) {
      std::string err = "Invalid species/temp for: " + pp_prefix;
      BoxLib::Abort(err.c_str());
    }
    measured_comps[0] = sCompY+comp;
  }
  num_measured_values = measurement_times.size() * measured_comps.size();

  Tfile = Tfile_DEF; pp.query("Tfile",Tfile);
  Patm = Patm_DEF; pp.query("Patm",Patm);

  measurement_error = CVReactorErr_DEF;
  pp.query("measurement_error",measurement_error);
}

CVReactor::CVReactor(const CVReactor& rhs)
  : cd(rhs.cd) {
  measurement_times = rhs.measurement_times;
  measured_comps = rhs.measured_comps;
  num_measured_values = num_measured_values;
  s_init.resize(rhs.s_init.box(),rhs.s_init.nComp()); s_init.copy(rhs.s_init);
  s_final.resize(rhs.s_final.box(),rhs.s_final.nComp()); s_final.copy(rhs.s_final);
  C_0.resize(rhs.C_0.box(),rhs.C_0.nComp()); C_0.copy(rhs.C_0);
  funcCnt.resize(rhs.funcCnt.box(),rhs.funcCnt.nComp()); funcCnt.copy(rhs.funcCnt);
  sCompY=rhs.sCompY;
  sCompT=rhs.sCompT;
  sCompR=rhs.sCompR;
  sCompRH=rhs.sCompRH;
  Patm=rhs.Patm;
}

void
CVReactor::GetMeasurementError(std::vector<Real>& observation_error)
{
  for (int i=0; i<NumMeasuredValues(); ++i) {
    observation_error[i] = measurement_error;
  }
}

void
CVReactor::GetMeasurements(std::vector<Real>& simulated_observations)
{
  BL_ASSERT(is_initialized);
  Reset();
  const Box& box = funcCnt.box();
  int Nspec = cd.numSpecies();
  int num_time_nodes = measurement_times.size();
  simulated_observations.resize(num_time_nodes);

#ifdef LMC_SDC
  FArrayBox& rYold = s_init;
  FArrayBox& rYnew = s_final;
  FArrayBox& rHold = s_init;
  FArrayBox& rHnew = s_final;
  FArrayBox& Told  = s_init;
  FArrayBox& Tnew  = s_final;
  FArrayBox* diag = 0;
#else
  FArrayBox& Yold = s_init;
  FArrayBox& Ynew = s_final;
  FArrayBox& Told = s_init;
  FArrayBox& Tnew = s_final;
#endif


  s_init.copy(s_save);
  s_final.copy(s_save);
  Real t_end = 0;
  for (int i=0; i<num_time_nodes; ++i) {
    Real t_start = t_end;
    t_end = measurement_times[i];
    Real dt = t_end - t_start;

#ifdef LMC_SDC
    cd.solveTransient_sdc(rYnew,rHnew,Tnew,rYold,rHold,Told,C_0,
                          funcCnt,box,sCompY,sCompRH,sCompT,
                          dt,Patm,diag,true);
    simulated_observations[i] = ExtractMeasurement();

    //std::cout << "Data :  " << t_end << " " << simulated_observations[i] << std::endl;

    rYold.copy(rYnew,sCompY,sCompY,Nspec);
    rHold.copy(rHnew,sCompRH,sCompRH,Nspec);
    Told.copy(Tnew,sCompT,sCompT,1);
#else
    cd.solveTransient(Ynew,Tnew,Yold,Told,funcCnt,box,
                      sCompY,sCompT,dt,Patm);
    simulated_observations[i] = ExtractMeasurement();

    Yold.copy(Ynew,sCompY,sCompY,Nspec);
    Told.copy(Tnew,sCompT,sCompT,1);
#endif
  }

  //std::cout << std::endl;
  //BoxLib::Abort();
}


Real
CVReactor::ExtractMeasurement() const
{
  // Return the final temperature of the cell that was evolved
  BL_ASSERT(is_initialized);
  return s_final(s_final.box().smallEnd(),measured_comps[0]);
}

void
CVReactor::Reset()
{
  if (is_initialized)
    funcCnt.setVal(0);
}

void
CVReactor::InitializeExperiment()
{
  std::ifstream is;
  is.open(pmf_file_name.c_str());
  FArrayBox fileFAB;
  fileFAB.readFrom(is);
  is.close();

  // Simple check to see if number of species is same between compiled mech and fab file
  const Box& box = fileFAB.box();
  const int nSpec = cd.numSpecies();
  const int nComp = nSpec + 4;
  if (nComp != fileFAB.nComp()) {
    std::cout << "pmf file is not compatible with the mechanism compiled into this code" << '\n';
    std::cout << "pmf file number of species: " << fileFAB.nComp() - 4 << '\n';
    std::cout << "expecting: " << nSpec << '\n';
    BoxLib::Abort();
  }

  // Find location
  bool found = false;
  IntVect iv=box.smallEnd();
  for (IntVect End=box.bigEnd(); iv<=End && !found; box.next(iv)) {
    if (fileFAB(iv,sCompT)>=Tfile) found = true;
  }

  Box bx(iv,iv);
  s_init.resize(bx,fileFAB.nComp()); s_init.copy(fileFAB);
  funcCnt.resize(bx,1);
  
#ifdef LMC_SDC
  s_init.mult(1.e3,sCompR,1);
  cd.getHmixGivenTY(s_init,s_init,s_init,bx,sCompT,sCompY,sCompRH);
  s_init.mult(s_init,sCompR,sCompRH,1);
  for (int i=0; i<nSpec; ++i) {
    s_init.mult(s_init,sCompR,sCompY+i,1);
  }
  C_0.resize(bx,nSpec+1); C_0.setVal(0);
#endif

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
  std::cerr << "Experiment " <<  pp_prefix  << std::endl;
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

void
PREMIXReactor::GetMeasurements(std::vector<Real>& simulated_observations)
{
  // This set to return a single value - the flame speed
  simulated_observations.resize(1);

  int lregrid;
  int lrstrt = 0;
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
                  std::cerr <<  "restart this time" << std::endl;
              }
              else{
                  (*pr)->lrstrtflag = 0;
                  lrstrt = 1; // restart on the next time through
  //                std::cerr <<  "restart next time" << std::endl;
              }
              std::vector<Real> pr_obs;
   //           std::cerr << " Running " << (*pr)->premix_input_file  << " with restart = " << (*pr)->lrstrtflag << std::endl;
              (*pr)->GetMeasurements(pr_obs);
    //          std::cerr << " Obtained intermediate observable " << pr_obs[0] << std::endl;
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
  
  // Extract the measurements - should probably put into an 'ExtractMeasurements'
  // for consistency with CVReactor
  if( *solsz > 0 ) {
    //std::cout << "Premix generated a viable solution " << std::endl;
    simulated_observations[0]  = savesol[*solsz + nmax*(ncomp-1)-1+3]; 
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
        std::cerr << "Initialized prereq " << i << " sz: " << (*pr)->nmax << std::endl;
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
