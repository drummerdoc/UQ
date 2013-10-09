#include <Rand.H>

// ******************************************************
// Generate uniformly distributed random number 
Real drand() {
  return (std::rand()+1.0)/(RAND_MAX+1.0);
}
  
// Generate standard normal random number 
Real randn(){
  Real pi; 
  pi =  3.14159265358979323846;
  return std::sqrt(-2*std::log(drand())) * std::cos(2*pi*drand());
}

