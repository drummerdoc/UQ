#include <UqPlotfile.H>

#include <FArrayBox.H>

#include <iostream>
#include <fstream>

void UqWriteSamples(const FArrayBox& fab, const std::string& name)
{
  std::ofstream ofs; ofs.open(name.c_str());
  fab.writeOn(ofs);
  ofs.close();

  std::cout << "data written to: " << name << std::endl;
}

void UqReadSamples(FArrayBox& fab, const std::string& name)
{
  std::ifstream ifs; ifs.open(name.c_str());
  fab.readFrom(ifs);
  ifs.close();
}

void UqPlotfileInfo( int *ndim, int *nwalkers, int *iters, const std::string& name)
{
  FArrayBox fab;
  UqReadSamples(fab,name);
  const Box& box = fab.box();
  *ndim = fab.nComp();
  *nwalkers = box.length(0);
  *iters = box.length(1);
}

void UqPlotfileRead( std::vector<double>& x, int walker, int iter, const std::string& name)
{
  FArrayBox fab;
  UqReadSamples(fab,name);
  const Box& box = fab.box();
  int ndim = fab.nComp();
  int nwalkers = box.length(0);
  int iters = box.length(1);

  if (walker >= nwalkers || iter >= iters) {
    BoxLib::Abort("walker or iter out of bounds");
  }

  IntVect iv(walker,iter);
  x.resize(ndim);
  for (int j=0; j<ndim; ++j) {
    x[j] = fab(iv,j);
  }

  std::cout << "samples read from: " << name << std::endl;
  for(int ii=0; ii<ndim; ii++){
    std::cout << x[ii] << std::endl;
  }
}


void UqPlotfileWrite( double x[], int ndim, int nwalkers, int iters, const std::string& name)
{
  /*
    Write a "plotfile" for UQ sampler calcs based on data passed into this routine,
    including the ensemble points (in "x"), and all the associated metadata.

    This is a proof-of-concept implementation not yet fleshed out, and thus only writes
    the ensemble points into a BoxLib "fab" data file.

    The incoming 3-index array x was flattened to 1D using
        index    = k + nwalkers*t + nwalkers*iters*j
        x[index] = x_3_index[k,t,j]
  */

  std::cout << "Creating a FAB on Box((0,0):(nwalkers-1,iters-1)) with components 0:ndim-1" << std::endl;

  Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(nwalkers-1,iters-1,0)));
  FArrayBox fab(box,ndim);

  for (int k=0; k<nwalkers; ++k) {
    for (int t=0; t<iters; ++t) {
      IntVect iv(k,t);
      for (int j=0; j<ndim; ++j) {
        long index = k + nwalkers*t + nwalkers*iters*j;
        fab(iv,j) = x[index];
      }
    }
  }

  UqWriteSamples(fab,name);
}

void UqPlotfileWrite(const std::vector<std::vector<double> >& x, const std::string& name)
{
  /*
    Write a "plotfile" for UQ sampler calcs based on data passed into this routine,
    including the ensemble points (in "x"), and all the associated metadata.

    This is a proof-of-concept implementation not yet fleshed out, and thus only writes
    the ensemble points into a BoxLib "fab" data file.

    The incoming 2-index array x is indexed x[iter][j]
  */
  int nwalkers = 1;
  int iters = x.size();
  if (iters == 0) {
    return;
  }
  int ndim = x[0].size();

  Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(nwalkers-1,iters-1,0)));
  FArrayBox fab(box,ndim);

  int k = 0; // walker
  for (int t=0; t<iters; ++t) {
    IntVect iv(k,t);
    for (int j=0; j<ndim; ++j) {
      fab(iv,j) = x[t][j];
    }
  }

  UqWriteSamples(fab,name);
}

