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


#endif // _SimulatedExperiment_
