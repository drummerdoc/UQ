#include <UqPlotfile.H>

#include <FArrayBox.H>
#include <Utility.H>
#include <ParallelDescriptor.H>

#include <iostream>
#include <fstream>

static const std::string PlotfileVersion = "UQ_Plotfile_V0";
static bool ioproc;

static void SetIOProc()
{
  ioproc = ParallelDescriptor::IOProcessor() || ParallelDescriptor::MyProc()<0;
}


UqPlotfile::UqPlotfile()
{
  SetIOProc();
}

UqPlotfile::UqPlotfile(const std::vector<double>& x,
                       int                        ndim,
                       int                        nwalkers,
                       int                        iter,
                       int                        iters,
                       const std::string&         rng_state)
  : m_ndim(ndim), m_nwalkers(nwalkers), m_iter(iter),
    m_iters(iters), m_rstate(rng_state)
{
  SetIOProc();

  Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(m_nwalkers-1,m_iters-1,0)));
  m_fab.resize(box,m_ndim);

  for (int k=0; k<m_nwalkers; ++k) {
    for (int t=0; t<m_iters; ++t) {
      IntVect iv(k,t);
      for (int j=0; j<m_ndim; ++j) {
        long index = k + m_nwalkers*t + m_nwalkers*m_iters*j;
        m_fab(iv,j) = x[index];
      }
    }
  }
}

std::vector<double>
UqPlotfile::LoadEnsemble(int iter, int iters) const
{
  BL_ASSERT(m_iters >= iter + iters);
  size_t len = m_nwalkers * m_ndim * iters;
  std::vector<double> result(len);
  for (int k=0; k<m_nwalkers; ++k) {
    for (int t=iter; t<iter+iters; ++t) {
      IntVect iv(k,t-m_iter); // Releative to my fab data
      for (int j=0; j<m_ndim; ++j) {
        long index = k + m_nwalkers*(t-iter) + m_nwalkers*iters*j;
        result[index] = m_fab(iv,j);
      }
    }
  }
  return result;
}

void
UqPlotfile::Write(const std::string& filename) const
{
  BuildDir(filename);
  WriteHeader(filename);
  WriteSamples(filename);
  WriteRState(filename);
}

void
UqPlotfile::Read(const std::string& filename)
{
  SetIOProc();
  ReadHeader(filename);
  ReadSamples(filename);
  ReadRState(filename);
}

void
UqPlotfile::WriteSamples(const std::string& filename) const
{
  if (ioproc) {
    std::ofstream ofs;
    ofs.open(DataName(filename).c_str());
    m_fab.writeOn(ofs);
    ofs.close();
  }
}

void
UqPlotfile::ReadSamples(const std::string& filename)
{
  if (ioproc) {
    std::ifstream ifs;
    ifs.open(DataName(filename).c_str());
    if (!ifs.good())
      BoxLib::FileOpenFailed(filename);
    m_fab.clear();
    m_fab.readFrom(ifs);
    ifs.close();

    const Box& box = m_fab.box();
    if (box.length(0) != m_nwalkers
        || box.length(1) != m_iters
        || m_fab.nComp() != m_ndim) {
      BoxLib::Abort("Bad UqPlofile");
    }
  }
  else {
    Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(m_nwalkers-1,m_iters-1,0)));
    m_fab.resize(box,m_ndim);
  }

  if (ParallelDescriptor::MyProc()>=0) {
    ParallelDescriptor::Bcast(m_fab.dataPtr(),m_fab.box().numPts());
  }
}

void
UqPlotfile::WriteHeader(const std::string& filename) const
{
  if (ioproc) {
    std::ofstream ofs;
    ofs.open(HeaderName(filename).c_str());
    ofs << PlotfileVersion << '\n';
    ofs << m_ndim << '\n';
    ofs << m_nwalkers << '\n';
    ofs << m_iter << '\n';
    ofs << m_iters << '\n';
    ofs.close();
  }
}

void
UqPlotfile::ReadHeader(const std::string& filename)
{
  int indata[4];

  if (ioproc) {
    std::ifstream ifs;
    ifs.open(HeaderName(filename).c_str());
    if (!ifs.good())
      BoxLib::FileOpenFailed(filename);
    std::string pfVersion;
    ifs >> pfVersion;
    BL_ASSERT(pfVersion == PlotfileVersion);

    ifs >> indata[0];
    ifs >> indata[1];
    ifs >> indata[2];
    ifs >> indata[3];
    ifs.close();
  }

  if (ParallelDescriptor::MyProc()>=0) {
    ParallelDescriptor::Bcast(indata,4);
  }

  m_ndim = indata[0];
  m_nwalkers = indata[1];
  m_iter = indata[2];
  m_iters = indata[3];
}

void
UqPlotfile::WriteRState(const std::string& filename) const
{
  if (ioproc) {
    std::ofstream ofs;
    ofs.open(RStateName(filename).c_str());
    if (m_rstate != "") 
      ofs << m_rstate;
    ofs.close();
  }
}

void
UqPlotfile::ReadRState(const std::string& filename)
{
  // if (ParallelDescriptor::IOProcessor()) {

  //   std::ifstream ifs(RStateName(filename).c_str());

  //   m_rstate = "";
  //   char c;
  //   while (ifs.get(c)) {
  //     m_rstate += c;
  //   }

  //   if (!ifs.eof()) {
  //     BoxLib::Abort("Error reading rstate");
  //   }
  //   ifs.close();
  // }
  for (int i=0; i<ParallelDescriptor::NProcs(); ++i) {
    if (i==ParallelDescriptor::MyProc()) {
      std::ifstream ifs(RStateName(filename).c_str());
      
      m_rstate = "";
      char c;
      while (ifs.get(c)) {
        m_rstate += c;
      }
      
      if (!ifs.eof()) {
        BoxLib::Abort("Error reading rstate");
      }
      ifs.close();
    }
  }
}

void
UqPlotfile::BuildDir(const std::string& filename) const
{
  if (ioproc)
    if (!BoxLib::UtilCreateDirectory(filename, 0755))
      BoxLib::CreateDirectoryFailed(filename);
}

std::string
UqPlotfile::HeaderName(const std::string& filename) const
{
  return filename + "/Header";
}

std::string
UqPlotfile::DataName(const std::string& filename) const
{
  return filename + "/Data.fab";
}

std::string
UqPlotfile::RStateName(const std::string& filename) const
{
  return filename + "/RState.pic";
}
