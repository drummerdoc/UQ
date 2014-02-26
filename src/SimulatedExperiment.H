#ifndef _SimulatedExperiment_
#define _SimulatedExperiment_

#include <ChemDriver.H>
#include <Rand.H>
#include <iostream>
#include <PremixSol.H>

typedef PArray<ChemDriver::Parameter> Parameters;

struct SimulatedExperiment
{
  SimulatedExperiment() : is_initialized(false) {}
  virtual ~SimulatedExperiment() {}
  virtual void GetMeasurements(std::vector<Real>& simulated_observations) = 0;
  virtual void GetMeasurementError(std::vector<Real>& observation_error) = 0;
  virtual int NumMeasuredValues() const = 0;
  virtual void InitializeExperiment() = 0;
  bool Initialized() const {return is_initialized;}
protected:
  bool is_initialized;
};


struct CVReactor
  : public SimulatedExperiment
{
  CVReactor(ChemDriver& _cd, const std::string& pp_prefix);
  CVReactor(const CVReactor& rhs);
  virtual ~CVReactor() {}

  virtual const std::vector<Real>& GetMeasurementTimes() const;
  virtual void GetMeasurements(std::vector<Real>& simulated_observations);
  virtual void GetMeasurementError(std::vector<Real>& observation_error);

  virtual void InitializeExperiment();

  virtual int NumMeasuredValues() const;

private:
  // Compute observation from evolution
  Real ExtractMeasurement() const;
  void Reset();

  ChemDriver& cd;

  std::vector<Real> measurement_times;
  std::vector<int> measured_comps;
  Real measurement_error;
  int num_measured_values;

  FArrayBox s_init, s_final, C_0, funcCnt, s_save; 
  int sCompY, sCompT, sCompR, sCompRH;
  Real Patm, Tfile;
  std::string pmf_file_name;
};

inline
int
CVReactor::NumMeasuredValues() const
{
  return num_measured_values;
}

inline
const std::vector<Real>&
CVReactor::GetMeasurementTimes() const
{
  return measurement_times;
}

extern "C" {
        void open_premix_files_(  int*, int*, int*, int*,int*, int *, int *,int*,int* );
        void close_premix_files_(  int*, int*, int*, int*,int* );
        void premix_( int*, int*, int*, int*, int*, int*,
                int*, int*, int*, int*, int*, double*, int* );
}

struct PREMIXReactor
  : public SimulatedExperiment
{
    PREMIXReactor(ChemDriver& _cd, const std::string& pp_prefix);

    virtual void GetMeasurements(std::vector<Real>& simulated_observations);
    virtual void GetMeasurementError(std::vector<Real>& observation_error);

    virtual void InitializeExperiment();
    int numComp() const;
    const PremixSol& getPremixSol() const;

    virtual int NumMeasuredValues() const;
    virtual ~PREMIXReactor();

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

    //
    // Sizes of data stored in object
    int maxsolsz; 
    int ncomp;

    ChemDriver& cd;
    PremixSol* premix_sol;
    Real measurement_error;
};

inline
int
PREMIXReactor::NumMeasuredValues() const
{
    // 1 value comes out - flame speed
    return 1;
}

#endif // _SimulatedExperiment_
