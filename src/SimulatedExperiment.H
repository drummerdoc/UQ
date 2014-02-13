#ifndef _SimulatedExperiment_
#define _SimulatedExperiment_

#include <ChemDriver.H>
#include <Rand.H>
#include <iostream>

typedef PArray<ChemDriver::Parameter> Parameters;

struct SimulatedExperiment
{
  SimulatedExperiment() {}
  SimulatedExperiment(const SimulatedExperiment& rhs);
  ~SimulatedExperiment() {}
  virtual const std::vector<Real>& GetMeasurementTimes() const = 0;
  virtual void GetMeasurements(std::vector<Real>& simulated_observations) = 0;
  virtual int NumMeasuredValues() const = 0;
  virtual void InitializeExperiment() = 0;
};


struct CVReactor
  : public SimulatedExperiment
{
  CVReactor(ChemDriver& _cd);
  CVReactor(const CVReactor& rhs);
  ~CVReactor() {}

  virtual const std::vector<Real>& GetMeasurementTimes() const;
  virtual void GetMeasurements(std::vector<Real>& simulated_observations);

  virtual void InitializeExperiment();

  virtual int NumMeasuredValues() const;

private:
  // Compute observation from evolution
  Real ExtractMeasurement() const;
  void Reset();

  ChemDriver& cd;

  std::vector<Real> measurement_times;
  std::vector<int> measured_comps;
  int num_measured_values;

  FArrayBox s_init, s_final, C_0, funcCnt, s_save; 
  int sCompY, sCompT, sCompR, sCompRH;
  Real Patm;
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
        void open_premix_files_(  int*, int*, int*, int*,int*, int *, int * );
        void close_premix_files_(  int*, int*, int*, int*,int* );
        void premix_( int*, int*, int*, int*, int*, int*,
                int*, int*, int*, int*, int*, double*, int* );
}

struct PREMIXReactor
  : public SimulatedExperiment
{
  virtual const std::vector<Real>& GetMeasurementTimes() const;
  virtual void GetMeasurements(std::vector<Real>& simulated_observations);

  virtual void InitializeExperiment();
  void setInputFile(char *);

  virtual int NumMeasuredValues() const;
    PREMIXReactor(ChemDriver& _cd);
//    PREMIXReactor();
    ~PREMIXReactor() {}
    
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

    void *iwork;
    void *rwork;
    void*cwork;
    void*lwork;

    // premix.inp file to read
    char inputfile[1024];
    //
    // Sizes of data stored in object
    int maxsolsz; 
    int ncomp;

    ChemDriver& cd;
    //
    //
    // This is a rather dumb object
    // Needs capacity of 
    // (maximum number of grid points)*(solution size)
    // where solution size includes the grid.
    // Inside premix copy
    // savesol[j] = X[j] where j=0,gridsize < jmax
    // savesol[j + (n*jmax)] where n = 1,natj; natj<maxstate)
    // Stores the premix solution to restart from
    //double * savesol;
    
};

inline
int
PREMIXReactor::NumMeasuredValues() const
{
    // 1 value comes out - flame speed
    return 1;
}

inline
const std::vector<Real>&
PREMIXReactor::GetMeasurementTimes() const
{
  std::cerr << "Not implmented GetMeasurementTimes()! \n";
  std::vector<Real> measurement_times;
  return measurement_times;
}


#endif // _SimulatedExperiment_
