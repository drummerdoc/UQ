#include <Driver.H>

static
void 
print_usage (int,
             char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0] << " pmf_file=<input fab file name> [options] \n";
  exit(1);
}

int
main (int   argc,
      char* argv[])
{
  BoxLib::Initialize(argc,argv);

  Driver driver;
  std::vector<double> data(1); data[0]=10000;
  //std::cout << driver.LogLikelihood(data) << std::endl;


  BoxLib::Finalize();
}

