#ifndef _Minimizer_H_
#define _Minimizer_H_

#include <REAL.H>

#include <vector>
#include <string>

typedef std::vector<std::vector<Real> > MyMat;

void writeHessian(const MyMat& H, const std::string& hessianOutFile);
MyMat readHessian(const std::string& hessianInFile);

class Minimizer
{
public:
  virtual bool minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln) = 0;
  virtual ~Minimizer() {}

  static MyMat FD_Hessian(void *p, const std::vector<Real>& X);
  static MyMat InvSqrt(void *p, const MyMat & H);
};

class GeneralMinimizer
  : public Minimizer
{
public:
  virtual bool minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln);
  virtual ~GeneralMinimizer() {}
};

class NLLSMinimizer
  : public Minimizer
{
public:
  virtual bool minimize(void *p, const std::vector<Real>& guess, std::vector<Real>& soln);
  virtual ~NLLSMinimizer() {}
  static MyMat  JTJ(void *p, const std::vector<Real>& X);
};

Real NegativeLogLikelihood(const std::vector<double>& parameters);

#endif
