#ifndef _UqPlotfile_H_
#define _UqPlotfile_H_

#include <string>
#include <vector>

void UqPlotfileWrite( double x[], int ndim, int nwalkers, int iters, const std::string& name);
void UqPlotfileWrite( const std::vector<std::vector<double> >& x, const std::string& name);

void UqPlotfileInfo( int *ndim, int *nwalkers, int *iters, const std::string& name);
void UqPlotfileRead( std::vector<double>& x, int walker, int iter, const std::string& name);

#endif
