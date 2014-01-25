%module pyemcee
%{
#  include <Driver.H>

#define SWIG_FILE_WITH_INIT
%}

%include "numpy.i"

%init %{
    import_array();
%}

%include <numpy.i>
%include <std_vector.i>

namespace std {
  %template(DoubleVec) std::vector<double>;
};

struct Driver
{
  Driver();
  ~Driver();
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();
};
