#ifndef _SimulatedExperiment_
#define _SimulatedExperiment_

#include <ChemDriver.H>
#include <Rand.H>
#include <iostream>
#include <PremixSol.H>

typedef PArray<ChemDriver::Parameter> Parameters;

struct SimulatedExperiment
{
  SimulatedExperiment();
  virtual ~SimulatedExperiment();

  typedef std::vector<std::string> ErrMap;
  static ErrMap& ErrorMap() {return err_map;}
  static const std::string& ErrorString(int errID);
  static int ErrorID(const std::string& errStr);

  virtual std::pair<bool,int> GetMeasurements(std::vector<Real>& simulated_observations) = 0;
  virtual void GetMeasurementError(std::vector<Real>& observation_error) = 0;
  virtual int NumMeasuredValues() const = 0;
  virtual void InitializeExperiment() = 0;
  bool Initialized() const {return is_initialized;}
  const std::string& DiagnosticName() const {return diagnostic_name;}
  const std::string& LogFile() const {return log_file;}
  virtual void CopyData(int src, int dest, int tag);
  int Verbosity() const {return verbosity;}
  void SetVerbosity(int verb) {verbosity = verb;}


protected:
  static ErrMap build_err_map();
  static ErrMap err_map;

  bool is_initialized;
  std::string diagnostic_name;
  std::string log_file;
  int verbosity;
};


struct ZeroDReactor
  : public SimulatedExperiment
{  
  enum REACTOR_TYPE
  {
    CONSTANT_VOLUME, 
    CONSTANT_PRESSURE
  };
  ZeroDReactor(ChemDriver& _cd, const std::string& pp_prefix, const REACTOR_TYPE& _type);
  virtual ~ZeroDReactor();

  virtual const std::vector<Real>& GetMeasurementTimes() const;
  virtual std::pair<bool,int> GetMeasurements(std::vector<Real>& simulated_observations);
  virtual void GetMeasurementError(std::vector<Real>& observation_error);

  virtual void InitializeExperiment();

  virtual int NumMeasuredValues() const;

  Real TransientThresh() const {return transient_thresh;}
  void ComputeMassFraction(FArrayBox& Y) const;

protected:
  // Compute observation from evolution
  Real ExtractMeasurement() const;
  void ExtractMeasurements(std::vector<Real>&, Real sample_time ) const;
  bool ValidMeasurement(Real data) const;
  void Reset();

  std::string name;
  ChemDriver& cd;
  REACTOR_TYPE reactor_type;

  std::vector<Real> measurement_times;
  std::vector<int> measured_comps;
  Real measurement_error;

// This is number of data points and number of values
// from simulation / experiment
  int num_measured_values;

  FArrayBox s_init, s_final, C_0, funcCnt, s_save; 
  int sCompY, sCompT, sCompR, sCompRH;
  Real Patm, Tfile;
  std::string pmf_file_name;
  Real transient_thresh;
  Real mean_delta_cond_start;
  Real mean_delta_cond_stop;

private:
  ZeroDReactor(const ZeroDReactor& rhs);
};

extern "C" {
  void open_premix_files_(  int*, int*, int*, int*, int*,int*, int *, int *,int*,int* );
  void close_premix_files_(  int*, int*, int*, int*,int* );
  void premix_( int*, int*, int*, int*, int*, int*,
                int*, int*, int*, int*, int*, double*, int* , int*,
                int *, int *, const int *,int *);
}

struct PREMIXReactor
  : public SimulatedExperiment
{
  PREMIXReactor(ChemDriver& _cd, const std::string& pp_prefix);

  virtual std::pair<bool,int> GetMeasurements(std::vector<Real>& simulated_observations);
  virtual void GetMeasurementError(std::vector<Real>& observation_error);

  virtual void CopyData(int src, int dest, int tag);

  virtual void InitializeExperiment();
  const PremixSol& getPremixSol() const;

  void solCopyIn( PremixSol * );
  void solCopyOut( PremixSol * );

  virtual int NumMeasuredValues() const;
  virtual ~PREMIXReactor();

  bool ValidMeasurement(Real data) const;

  std::string name;

  // Array of PREMIXReactors that are prereqs to get this solution
  Array<PREMIXReactor*> prereq_reactors;

  // Sizes for work arrays
  int lenlwk;
  int leniwk;
  int lenrwk;
  int lencwk;
  int lensym;

  // Unit numbers for input/output files
  int lin;
  int lout;
  int lrin;
  int lrout;
  int lrcvr;
  int linck;
  int linmc;

  // Pass this as maximum number of gridpoints
  int nmax;

  void * iwork;
  void * rwork;
  void * cwork;
  void * lwork;

  // premix.inp file to read
  std::string premix_input_file;
  std::string premix_input_path;
  std::string lmc_soln_file;

  int lrstrtflag;

  ChemDriver& cd;
  PremixSol* premix_sol;
  Real measurement_error;

  int max_premix_iters;
};

inline
int
PREMIXReactor::NumMeasuredValues() const
{
  // 1 value comes out - flame speed
  return 1;
}

#endif // _SimulatedExperiment_
