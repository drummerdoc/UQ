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
#ifdef BL_USE_MPI
  void SetComm(MPI_Comm mpi_comm);
#endif
  ~Driver();
  void SetParallelModeThreaded();
  void SetNumThreads(int num_threads);

  static double LogLikelihood(const std::vector<double>& parameters);
  static double VerboseLogLikelihood(const std::vector<Real>& pvals,
				     std::vector<Real>&       dvals,
				     std::vector<Real>&       svals,
				     Real&                    Fa,
				     Real&                    Fb);

  static int NumParams();
  static int NumData();
  static std::vector<double> PriorMean();
  static std::vector<double> TrueParameters();
  static std::vector<double> PriorStd();
  static std::vector<double> LowerBound();
  static std::vector<double> UpperBound();
  static std::vector<double> EnsembleStd();
  static std::vector<double> GenerateTestMeasurements(const std::vector<Real>& test_params);

  static const std::vector<Real>& MeasuredDataSTD();
  static const std::vector<Real>& MeasuredData();

  static ChemDriver *cd;
  static MINPACKstruct *mystruct;
  static Real param_eps;
  bool mpi_initialized;
  int omp_threads_override;

  MPI_Comm _mpi_comm;

};

#endif
