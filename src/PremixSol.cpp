
#include <PremixSol.H>

#include <fstream>

#include <Utility.H>
#include <FArrayBox.H>

static std::string CheckPointVersion = "PremixSol_V1";

bool
PremixSol::WriteSoln (const std::string& filename) const
{
  if( ! BoxLib::UtilCreateDirectory(filename, 0755)) {
    BoxLib::CreateDirectoryFailed(filename);
  }
  std::string HeaderFileName = filename + "/Header";

  std::ofstream HeaderFile;
  HeaderFile.open(HeaderFileName.c_str(), std::ios::out | std::ios::trunc |
		  std::ios::binary);

  if ( ! HeaderFile.good())
    BoxLib::FileOpenFailed(HeaderFileName);

  int old_prec = HeaderFile.precision(17);

  HeaderFile << CheckPointVersion << '\n'
	     << ncomp             << '\n'
	     << maxgp             << '\n'
	     << nextra            << '\n'
	     << ngp               << '\n';

  Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(ngp,0,0))); // Last point used to hold 2 extra quantities
  FArrayBox fab(box,ncomp);
  for (int j=0; j<ngp; ++j) {
    IntVect iv(D_DECL(j,0,0));
    for (int n=0; n<ncomp; ++n) {
      fab(iv,n) = solvec[ j + n * maxgp ];
    }
  }
  IntVect iv(D_DECL(ngp,0,0));
  fab(iv,0) = solvec[ngp + (ncomp-1)*maxgp    ]; // Pcgs;
  fab(iv,1) = solvec[ngp + (ncomp-1)*maxgp + 1]; // rho.u at inlet
	  
  HeaderFile.precision(old_prec);

  std::string DataFileName = filename + "/Data.fab";

  std::ofstream DataFile;
  DataFile.open(DataFileName.c_str(), std::ios::out | std::ios::trunc |
		std::ios::binary);

  if ( ! DataFile.good())
    BoxLib::FileOpenFailed(DataFileName);

  fab.writeOn(DataFile);

  if ( ! HeaderFile.good() || ! DataFile.good()) {
    std::cout << "PremixSol::WriteSoln() failed: " << filename << std::endl;
    return false;
  }
  return true;
}

bool
PremixSol::ReadSoln (const std::string& filename)
{
  std::string HeaderFileName = filename + "/Header";

  std::ifstream HeaderFile;
  HeaderFile.open(HeaderFileName.c_str(), std::ios::in | std::ios::binary);

  if ( ! HeaderFile.good())
    BoxLib::FileOpenFailed(HeaderFileName);

  std::string version;
  HeaderFile >> version;
  if (version != CheckPointVersion) {
    std::cout << "Bad format for premixsol restart file: " << version << std::endl;;
    BoxLib::Abort();
  }

  HeaderFile >> ncomp;
  HeaderFile >> maxgp;
  HeaderFile >> nextra;
  HeaderFile >> ngp;

  FArrayBox fab;
  std::string DataFileName = filename + "/Data.fab";

  std::ifstream DataFile;
  DataFile.open(DataFileName.c_str(), std::ios::in | std::ios::binary);

  if ( ! DataFile.good())
    BoxLib::FileOpenFailed(DataFileName);

  Box box(IntVect(D_DECL(0,0,0)),IntVect(D_DECL(ngp,0,0))); // Last point used to hold 2 extra quantities
  fab.readFrom(DataFile);
  BL_ASSERT(box == fab.box());
  BL_ASSERT(ncomp == fab.nComp());

  delete solvec;
  solvec = new double[ncomp*maxgp + nextra];

  for (int j=0; j<ngp; ++j) {
    IntVect iv(D_DECL(j,0,0));
    for (int n=0; n<ncomp; ++n) {
      solvec[ j + n * maxgp ] = fab(iv,n);
    }
  }
  IntVect iv(D_DECL(ngp,0,0));
  solvec[ngp + (ncomp-1)*maxgp    ] = fab(iv,0); // Pcgs;
  solvec[ngp + (ncomp-1)*maxgp + 1] = fab(iv,1); // rho.u at inlet

  if ( ! DataFile.good()) {
    std::cout << "PremixSol::ReadSoln() datafile read failed: " << filename << std::endl;
    return false;
  }
  return true;
}
