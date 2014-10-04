
#include <Minimizer.H>

class Sampler
{
public:
  // Sample
  virtual void Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const = 0;

  // Normalize weights to that their sum is 1
  void NormalizeWeights(std::vector<Real>& w) const;

  // Compute the mean of a scalar from samples
  void ScalarMean(Real& Mean, std::vector<Real> & samples) const;

  // Compute the mean of a vector from samples
  void Mean(std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples) const;

  // Compute the weighted mean of a vector from samples
  void WeightedMean(std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples) const;

  // Compute the covariance matrix from samples
  void Var(std::vector<Real>& Var,std::vector<Real>& Mean, std::vector<std::vector<Real> >& samples) const;

  // Compute covariance from weighted samples
  void WeightedVar(std::vector<Real>& CondVar,std::vector<Real>& CondMean, std::vector<Real>& w, std::vector<std::vector<Real> >& samples) const;

  // Compute quality measure of ensemble
  Real CompR(std::vector<Real>& w, int NOS) const;

  // Approximate effective sample size
  Real EffSampleSize(std::vector<Real>& w, int NOS) const;

  // Write samples and weights to file
  void WriteSamplesWeights(std::vector<std::vector<Real> >& samples, std::vector<Real>& w, const char *tag) const;

  // Resampling
  void Resampling(std::vector<std::vector<Real> >& Xrs,std::vector<Real>& w,std::vector<std::vector<Real> >& samples) const;

  // Write samples to file
  void WriteResampledSamples(std::vector<std::vector<Real> >& Xrs, const char *tag) const;

  // Quadratic approximation of F
  Real F0(const std::vector<Real>& sample, const std::vector<Real>& mu, const MyMat& H, Real phi) const;

  virtual ~Sampler() {}
};

// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////

class PriorMCSampler
  : public Sampler
{
/*
  This is a weighted sampler
  It produces samples from prior and weights them with
  the likelihood. 
  It will work well in problems where the prior is close to
  the posterior
*/
public:

  PriorMCSampler(const std::vector<Real>& prior_mean,
                 const std::vector<Real>& prior_std);

  virtual void Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const;

  virtual ~PriorMCSampler() {}

protected:
  std::vector<Real> prior_mean;
  std::vector<Real> prior_std;
};

// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////

class LinearMapSampler
  : public Sampler
{
/*
  The linear map sampler (implicit sampling)
*/
public:
  LinearMapSampler(const std::vector<Real>& mu,
                   const MyMat& H,
                   const MyMat& invsqrt,
                   Real phi);

  virtual void Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const;

  virtual ~LinearMapSampler() {}

protected:
  std::vector<Real> mu;
  const MyMat& H, invsqrt;
  Real phi;
};

// /////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////

class SymmetrizedLinearMapSampler
  : public LinearMapSampler
{
/*
  Symmetrized linear map sampler (implicit sampling)
*/
public:
  SymmetrizedLinearMapSampler(const std::vector<Real>& mu,
                              const MyMat& H,
                              const MyMat& invsqrt,
                              Real phi)
    : LinearMapSampler(mu,H,invsqrt,phi) {}

  virtual void Sample(void* p, std::vector<std::vector<Real> >& samples, std::vector<Real>& w) const;

  virtual ~SymmetrizedLinearMapSampler() {}
};


