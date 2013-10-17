#include <fstream>
#include <iostream>

#include <SimulatedExperiment.H>
#include <ParmParse.H>

static Real Patm_DEF = 1;
static Real dt_DEF   = 0.1;
static Real Tfile_DEF = 900;
static int  num_time_intervals_DEF = 10;

CVReactor::CVReactor(ChemDriver& _cd)
  : cd(_cd)
{
  ParmParse pp;
  Real dt = dt_DEF; pp.query("dt",dt);

  int num_time_intervals = num_time_intervals_DEF;
  pp.query("time_intervals",num_time_intervals);
  measurement_times.resize(num_time_intervals);
  for (int i=0; i<num_time_intervals; ++i) {
    measurement_times[i] = (i+1)*dt/num_time_intervals;
  }
  sCompT  = 1;
  sCompRH = 2;
  sCompR  = 3;
  sCompY  = 4;

  bool do_temp = true; pp.query("do_temp",do_temp);
  measured_comps.resize(1);
  if (do_temp) {
    measured_comps[0] = sCompT;
  }
  else {
    measured_comps[0] = sCompY+cd.index("OH");
  }
  num_measured_values = measurement_times.size() * measured_comps.size();

}

CVReactor::CVReactor(const CVReactor& rhs)
  : cd(rhs.cd) {
  measurement_times = rhs.measurement_times;
  measured_comps = rhs.measured_comps;
  num_measured_values = num_measured_values;
  s_init.resize(rhs.s_init.box(),rhs.s_init.nComp()); s_init.copy(rhs.s_init);
  s_final.resize(rhs.s_final.box(),rhs.s_final.nComp()); s_final.copy(rhs.s_final);
  C_0.resize(rhs.C_0.box(),rhs.C_0.nComp()); C_0.copy(rhs.C_0);
  I_R.resize(rhs.I_R.box(),rhs.I_R.nComp()); I_R.copy(rhs.I_R);
  funcCnt.resize(rhs.funcCnt.box(),rhs.funcCnt.nComp()); funcCnt.copy(rhs.funcCnt);
  sCompY=rhs.sCompY;
  sCompT=rhs.sCompT;
  sCompR=rhs.sCompR;
  sCompRH=rhs.sCompRH;
  Patm=rhs.Patm;
}

void
CVReactor::GetMeasurements(Array<Real>& simulated_observations)
{
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
    cd.solveTransient_sdc(rYnew,rHnew,Tnew,rYold,rHold,Told,C_0,I_R,
                          funcCnt,box,sCompY,sCompRH,sCompT,
                          dt,Patm,diag,true);
    simulated_observations[i] = ExtractMeasurement();
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
}


Real
CVReactor::ExtractMeasurement() const
{
  // Return the final temperature of the cell that was evolved
  //return s_final(s_final.box().smallEnd(),sCompT);
  return s_final(s_final.box().smallEnd(),measured_comps[0]);
}

void
CVReactor::Reset()
{
  funcCnt.setVal(0);
}

void
CVReactor::InitializeExperiment()
{
  ParmParse pp;
  std::string pmf_file=""; pp.get("pmf_file",pmf_file);
  std::ifstream is;
  is.open(pmf_file.c_str());
  FArrayBox fileFAB;
  fileFAB.readFrom(is);
  is.close();

  // Simple check to see if number of species is same between compiled mech and fab file
  const Box& box = fileFAB.box();
  const int nSpec = cd.numSpecies();
  const int nComp = nSpec + 4;
  if (nComp != fileFAB.nComp()) {
    std::cout << "pmf file is not compatible with the mechanism compiled into this code" << '\n';
    BoxLib::Abort();
  }

  // Find location
  bool found = false;
  IntVect iv=box.smallEnd();
  Real Tfile = Tfile_DEF; pp.query("Tfile",Tfile);
  for (IntVect End=box.bigEnd(); iv<=End && !found; box.next(iv)) {
    if (fileFAB(iv,sCompT)>=Tfile) found = true;
  }

  Box bx(iv,iv);
  Patm = Patm_DEF; pp.query("Patm",Patm);
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
  I_R.resize(bx,nSpec+1); I_R.setVal(0);
#endif

  s_final.resize(bx,s_init.nComp());
  s_final.copy(s_init);

  s_save.resize(bx,s_init.nComp());
  s_save.copy(s_init);
}

