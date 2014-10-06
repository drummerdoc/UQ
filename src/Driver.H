#ifndef _Driver_H_
#define _Driver_H_

#include <ExperimentManager.H>
#include <MINPACKstruct.H>
#include <ParallelDescriptor.H>

struct Driver
{
  Driver(int, char **, int mpi_later);
  // Driver(int, char **, MPI_Comm mpi_comm);
  void init(int, char **);
  void SetComm(MPI_Comm mpi_comm);
  ~Driver();
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();
  static int NumData();
  static std::vector<double> PriorMean();
  static std::vector<double> PriorStd();
  static std::vector<double> EnsembleStd();
  static std::vector<double> GenerateTestMeasurements(const std::vector<Real>& test_params);

  static ChemDriver *cd;
  static MINPACKstruct *mystruct;
  static Real param_eps;
  bool mpi_initialized;

  MPI_Comm _mpi_comm;

};

#endif
