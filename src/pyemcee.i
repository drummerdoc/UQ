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
  Driver(int argc, char**argv );
  ~Driver();
  static double LogLikelihood(const std::vector<double>& parameters);
  static int NumParams();
  static int NumData();
  static std::vector<double> PriorMean();
  static std::vector<double> PriorStd();
  static std::vector<double> GenerateTestMeasurements(const std::vector<double>& test_params);
};
