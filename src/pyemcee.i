%module pyemcee
// This tells SWIG to treat char ** as a special case
// Shamelessly lifted from http://www.swig.org/Doc1.1/HTML/Python.html
%typemap(python,in) char ** {
    /* Check if is a list */
    if (PyList_Check($source)) {
        int size = PyList_Size($source);
        int i = 0;
        $target = (char **) malloc((size+1)*sizeof(char *));
        for (i = 0; i < size; i++) {
            PyObject *o = PyList_GetItem($source,i);
            if (PyString_Check(o))
                $target[i] = PyString_AsString(PyList_GetItem($source,i));
            else {
                PyErr_SetString(PyExc_TypeError,"list must contain strings");
                free($target);
                return NULL;
            }
        }
        $target[i] = 0;
    } else {
        PyErr_SetString(PyExc_TypeError,"not a list");
        return NULL;
    }
}
%{
#  include <Driver.H>
#  include <ParmParse.H>

#define SWIG_FILE_WITH_INIT


    %}

    %include "numpy.i"
#ifdef BL_USE_MPI
    %include "mpi4py.i"
    %mpi4py_typemap(Comm, MPI_Comm);
#endif

    %init %{
        import_array();
        %}

        %include <numpy.i>
        %include <std_vector.i>
        %include <std_string.i>

        namespace std {
            %template(DoubleVec) std::vector<double>;
            %template(StringVec) std::vector<std::string>;
        };

struct Driver
{
  Driver(int argc, char**argv, int mpi_later);
  ~Driver();
  void init(int argc, char**argv);
#ifdef BL_USE_MPI
  void SetComm(MPI_Comm comm);
#endif
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();
  static int NumData();
  static std::vector<double> PriorMean();
  static std::vector<double> PriorStd();
  static std::vector<double> EnsembleStd();
  static std::vector<double> LowerBound();
  static std::vector<double> UpperBound();
  static std::vector<double> GenerateTestMeasurements(const std::vector<double>& test_params);
};

/* Here, we expose the BoxLib::ParmParse class, but only for strings, lists of strings

   Initialize/use as:

   >>> pp = pyemcee.ParmParse()
   >>> pname = 'myparam'
   >>> n = pp.count(pname)
   >>> if n == 1:
   >>>   print pname + ' = ' + pp[pname]  # Gets a single parameter value
   >>> elif n > 1:
   >>>   p = pp.getarr(pname)             # Gets a list of parameter values
   >>>   print pname+' (' + str(n) + '):'
   >>>   for pi in p:
   >>>     print('  p: '+pi)

 */ 
struct ParmParse
{
  %extend{
    const std::string __getitem__(const std::string& name) {
      std::string result;
      self->get(name.c_str(),result);
      return result.c_str();
    }

    const std::vector<std::string> getarr(const std::string& name) {
      std::vector<std::string> result;
      int n = self->countval(name.c_str());
      self->getarr(name.c_str(),result,0,n);
      return result;
    }

    int count(const std::string& name) {
      return self->countval(name.c_str());
    }
  }
};
