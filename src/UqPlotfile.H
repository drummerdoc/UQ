#ifndef _UqPlotfile_H_
#define _UqPlotfile_H_

#include <string>
#include <vector>

#include <FArrayBox.H>

struct UqPlotfile
{
  UqPlotfile() {}
  UqPlotfile(const std::vector<double>& x, int ndim, int nwalkers, int iter, int iters, const std::string& rstate);

  void Write(const std::string& filename) const;
  void Read(const std::string& filename);

  std::vector<double> LoadEnsemble(int iter, int iters) const;

  int NDIM() const {return m_ndim;}
  int NWALKERS() const {return m_nwalkers;}
  int NITERS() const {return m_iters;}
  int ITER() const {return m_iter;}
  std::string RSTATE() const {return m_rstate;}

private:

  void WriteSamples(const std::string& filename) const;
  void ReadSamples(const std::string& filename);
  void WriteHeader(const std::string& filename) const;
  void ReadHeader(const std::string& filename);
  void WriteRState(const std::string& filename) const;
  void ReadRState(const std::string& filename);
  void BuildDir(const std::string& filename) const;

  std::string HeaderName(const std::string& filename) const;
  std::string DataName(const std::string& filename) const;
  std::string RStateName(const std::string& filename) const;

  FArrayBox m_fab;
  int m_ndim, m_nwalkers, m_iter, m_iters;
  std::string m_rstate;
};


#endif
