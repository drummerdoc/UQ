#ifndef _SimulatedExperiment_
#define _SimulatedExperiment_

#include <ChemDriver.H>
#include <Rand.H>

typedef PArray<ChemDriver::Parameter> Parameters;

struct SimulatedExperiment
{
  SimulatedExperiment() {}
  SimulatedExperiment(const SimulatedExperiment& rhs);
  ~SimulatedExperiment() {}
  virtual void GetMeasurements(Array<Real>& simulated_observations) = 0;
  virtual int NumMeasuredValues() const = 0;
  virtual void InitializeExperiment() = 0;
};


struct CVReactor
  : public SimulatedExperiment
{
  CVReactor(ChemDriver& _cd);
  CVReactor(const CVReactor& rhs);
  ~CVReactor() {}
  virtual void GetMeasurements(Array<Real>& simulated_observations);
  int NumMeasuredValues() const {return num_measured_values;}
  virtual void InitializeExperiment();

private:
  // Compute observation from evolution
  Real ExtractMeasurement() const;
  void Reset();

  ChemDriver& cd;

  Array<Real> measurement_times;
  Array<int> measured_comps;
  int num_measured_values;

  FArrayBox s_init, s_final, C_0, I_R, funcCnt, s_save; 
  int sCompY, sCompT, sCompR, sCompRH;
  Real Patm;
};

#endif // _SimulatedExperiment_
