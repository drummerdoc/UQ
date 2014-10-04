#ifndef _Minimizer_H_
#define _Minimizer_H_

#include <REAL.H>

#include <vector>

#define real __cminpack_real__

typedef std::vector<std::vector<Real> > MyMat;

#define CEN_DIFF
#undef FWD_DIFF


class Minimizer
{
public:
  virtual void minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln) = 0;
  virtual ~Minimizer() {}

  static MyMat FD_Hessian(void *p, const std::vector<Real>& X);
  static MyMat InvSqrt(void *p, const MyMat & H, const Real CutOff);
};

class GeneralMinimizer
  : public Minimizer
{
public:
  virtual void minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln);
  virtual ~GeneralMinimizer() {}
};

class NLLSMinimizer
  : public Minimizer
{
public:
  virtual void minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln);
  virtual ~NLLSMinimizer() {}
};

Real NegativeLogLikelihood(const std::vector<double>& parameters);

#endif
