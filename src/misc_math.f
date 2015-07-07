C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKCOPY (NN, X1, X2)
C
C  START PROLOGUE
C  SUBROUTINE CKCOPY (NN, X1, X2)
C  Copy X1(*) array members into X2(*) array.
C
C  INPUT
C  NN        - Integer scalar; number of elements to copy.
C  X1(*)     - Real array; dimension at least NN.
C
C  OUTPUT
C  X2(*)     - Real array; dimension at least NN.
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      DIMENSION X1(NN), X2(NN)
      DO 10 N = 1, NN
         X2(N) = X1(N)
   10 CONTINUE
C
C     end of CKCOPY
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKAVG (NN, S1, S2, SAVG)
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER(I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER(I-N)
C*****END precision > single
C
C     START PROLOGUE
C     For arrays of length nn,
C     SAVG(n) is the average value of S1(n) and S2(n).
C     END PROLOGUE
C
      DIMENSION S1(NN), S2(NN), SAVG(NN)
C
      DO 10 N = 1, NN
         SAVG(N) = 0.5 * (S1(N) + S2(N))
   10 CONTINUE
C
C     end of SUBROUTINE CKAVG
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKSNUM (LINE, NEXP, LOUT, KRAY, NN, KNUM, NVAL,
     1                   RVAL, KERR)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSNUM (LINE, NEXP, LOUT, KRAY, NN, KNUM, NVAL,
C                     RVAL, KERR)
C  Search a character string, LINE, for (1) a character substring which
C  may also appear in an array of character substrings KRAY, and
C  (2) some number of character substrings representing numbers.
C  In the case of (1), if the character substring appears in KRAY,
C  KNUM is its index position.
C  In the case of (2), the character substrings are converted to
C  NVAL real numbers and stored in RVAL, until NEXP are converted.
C
C  This allows format-free input of combined alpha-numeric data.
C  For example, the subroutine might be called to find a Chemkin
C  species index and convert the other substrings to real values:
C
C     input:  LINE    = "N2  1.2"
C             NEXP    = 1, the number of values expected
C             LOUT    = 6, a logical unit number on which to write
C                       diagnostic messages.
C             KRAY(*) = "H2" "O2" "N2" "H" "O" "N" "OH" "H2O" "NO"
C             NN      = 9, the number of entries in KRAY(*)
C     output: KNUM    = 3, the index number of the substring in
C                       KRAY(*) which corresponds to the first
C                       substring in LINE
C             NVAL    = 1, the number of values found in LINE
C                       following the first substring
C             RVAL(*) = 1.200E+00, the substring converted to a number
C             KERR    = .FALSE.
C  INPUT
C  LINE      - Character string; length depends on calling routine.
C  NEXP      - Integer scalar, number of values to be found in LINE.
C              If NEXP < 0, then IABS(NEXP) values are expected, but
C              it is not an error condition if less values are found.
C  LOUT      - Integer scalar, formatted output file unit.
C  KRAY(*)   - Character string array.
C  NN        - Integer scalar, total number of character strings
C              in KRAY.
C
C  OUTPUT
C  KNUM      - Integer scalar, index of character string in KRAY
C              which corresponds to the first substring in LINE.
C  NVAL      - Integer scalar, count of real values found in LINE.
C  RVAL(*)   - Real array, real values found in LINE; dimension at least
C              NEXP.
C  KERR      - Logical, syntax or dimensioning error flag;
C              corresponding string not found, or total of
C              values found is not the number of values expected,
C              will result in KERR = .TRUE.
C
C  END PROLOGUE
C     A '!' will comment out a line, or remainder of the line.
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*), KRAY(*)*(*), ISTR*80
      DIMENSION RVAL(*)
      LOGICAL KERR, IERR
      INTEGER CKFRCH, CKLSCH, CKSLEN
      EXTERNAL CKFRCH, CKLSCH, CKSLEN
C
      NVAL = 0
      KERR = .FALSE.
      ILEN = CKSLEN(LINE)
      IF (ILEN .LE. 0) RETURN
C
      I1 = CKFRCH(LINE(1:ILEN))
      I3 = INDEX(LINE(I1:ILEN),' ')
      IF (I3 .EQ. 0) I3 = ILEN - I1 + 1
      I2 = I1 + I3
      ISTR = ' '
      ISTR = LINE(I1:I2-1)
C
      CALL CKCOMP (ISTR, KRAY, NN, KNUM)
      IF (KNUM.EQ.0) THEN
         LT = MAX (CKLSCH(ISTR), 1)
         WRITE (LOUT,'(A)')
     1   ' Error in CKSNUM...'//ISTR(1:LT)//' not found...'
         KERR = .TRUE.
      ENDIF
C
      ISTR = ' '
      ISTR = LINE(I2:ILEN)
      IF (NEXP .NE. 0)
     1      CALL CKXNUM (ISTR, NEXP, LOUT, NVAL, RVAL, IERR)
C
C     end of SUBROUTINE CKSNUM
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKLEN (LINC, LOUT, LI, LR, LC, IFLAG)
C
C  START PROLOGUE
C
C  SUBROUTINE CKLEN (LINC, LOUT, LENI, LENR, LENC, IFLAG)
C   Returns the lengths required for work arrays.
C
C  INPUT
C  LINC     - Integer scalar, input file unit for the linkfile.
C  LOUT     - Integer scalar, formatted output file unit.
C
C  OUTPUT
C  LENI     - Integer scalar, minimum length required for the
C             integer work array.
C  LENR     - Integer scalar, minimum length required for the
C             real work array.
C  LENC     - Integer scalar, minimum length required for the
C             character work array.
C  IFLAG    - Integer scalar, indicates successful reading of
C             linkfile; IFLAG>0 indicates error type.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      PARAMETER (NLIST = 1)
      LOGICAL KERR, VOK, POK, LBIN
      CHARACTER*16 LIST(NLIST), FILVER, PREC, PRVERS, IFMT, CFMT,
     1             RFMT, LFMT
      PARAMETER (IFMT='(10I12)', CFMT='(8A16)', RFMT='(1P,5E24.16)',
     1           LFMT='(L8)')
C
      include 'ckcom.fh'
      DATA LIST(1) /'1.0'/
C
      FILVER = ' '
      PRVERS   = ' '
      PREC = ' '
      LENI = 0
      LENR = 0
      LENC = 0
C
      KERR = .FALSE.
      IFLAG = 0
      REWIND LINC
      NREC = 1
C*****linkfile (gas) > binary
C      LBIN = .TRUE.
C*****END linkfile (gas) > binary
C*****linkfile (gas) > ascii
      LBIN = .FALSE.
C*****END linkfile (gas) > ascii
C
      IF (LBIN) THEN
         READ (LINC, ERR=100) FILVER
      ELSE
         READ (LINC, CFMT, ERR=100) FILVER
      ENDIF
      CALL CKCOMP (FILVER, LIST, NLIST, IND)
      IF (IND .LE. 0) THEN
         VOK = .FALSE.
      ELSE
         VOK = .TRUE.
      ENDIF
C
      IF (LBIN) THEN
         NREC = 2
         READ (LINC, ERR=100) PRVERS
         NREC = 3
         READ (LINC, ERR=100) PREC
         NREC = 4
         READ (LINC, ERR=100) KERR
      ELSE
         NREC = 2
         READ (LINC, CFMT, ERR=100) PRVERS
         NREC = 3
         READ (LINC, CFMT, ERR=100) PREC
         NREC = 4
         READ (LINC, LFMT, ERR=100) KERR
      ENDIF
C
      POK = .FALSE.
C*****precision > double
      IF (INDEX(PREC, 'DOUB') .GT. 0) POK = .TRUE.
C*****END precision > double
C*****precision > single
C      IF (INDEX(PREC, 'SING') .GT. 0) POK = .TRUE.
C*****END precision > single
C
      IF (KERR .OR. (.NOT.POK) .OR. (.NOT.VOK)) THEN
         IF (KERR) THEN
            WRITE (LOUT,'(/A,/A)')
     1      ' There is an error in the Chemkin linkfile...',
     2      ' Check CHEMKIN INTERPRETER output for error conditions.'
         ENDIF
         IF (.NOT. VOK) WRITE (LOUT, '(/A)')
     1   ' Chemkin linkfile is incompatible with Chemkin-III Library'
         IF (.NOT. POK) THEN
            WRITE (LOUT,'(/A,A)')
     1      ' Precision of Chemkin linkfile does not agree with',
     2      ' precision of Chemkin library'
         ENDIF
         IFLAG = 20
         REWIND LINC
         RETURN
      ENDIF
C
      NREC = 5
      IF (LBIN) THEN
         READ (LINC, ERR=100) LENICK, LENRCK, LENCCK
      ELSE
         READ (LINC, IFMT, ERR=100) LENICK, LENRCK, LENCCK
      ENDIF
      REWIND LINC
C
      LENI = LENICK
      LENR = LENRCK
      LENC = LENCCK
      LI   = LENI
      LR   = LENR
      LC   = LENC
      RETURN
C
  100 CONTINUE
      IFLAG = NREC
      WRITE (LOUT, '(/A,/A,I5)')
     1   ' Error reading linkfile,',
     2   ' SUBROUTINE CKLEN record index #', IFLAG
      REWIND LINC
C
C     Generic Formats - limit lines to 132 characters
C
c8001  FORMAT (10I12)
c8002  FORMAT (1P,5E24.16)
c8003  FORMAT (8A16)
C
C     end of SUBROUTINE CKLEN
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKCOMP (IST, IRAY, II, I)
C
C  START PROLOGUE
C
C  SUBROUTINE CKCOMP (IST, IRAY, II, I)*
C  Returns the index of an element of a reference character string
C  array which corresponds to a character string;
C  leading and trailing blanks are ignored.
C
C
C  INPUT
C  IST      - Character string; length determined by application
C             program.
C  IRAY(*)  - Character string array; dimension at least II, the total
C             number of character strings for be searched.
C  II       - Integer scalar, the length of IRAY to be searched.
C
C  OUTPUT
C  I        - Integer scalar, the first array index in IRAY of a
C             character string IST, or 0 if IST is not found.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER*(*) IST, IRAY(*)
      INTEGER CKLSCH, CKFRCH
      EXTERNAL CKLSCH, CKFRCH
C
      I = 0
      IS1 = CKFRCH(IST)
      IS2 = CKLSCH(IST)
      ISLEN = IS2 - IS1 + 1
      ILEN = LEN(IRAY(1))
      IF (ILEN .LT. ISLEN) RETURN
      DO 10 N = 1, II
         IR1 = CKFRCH(IRAY(N)(1:ILEN))
         IR2 = CKLSCH(IRAY(N)(1:ILEN-1))
         IF (IR2-IR1+1 .NE. ISLEN) GO TO 10
         IF (IST(IS1:IS2) .EQ. IRAY(N)(IR1:IR2)) THEN
            I = N
            RETURN
         ENDIF
   10 CONTINUE
C
C     end of SUBROUTINE CKCOMP
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKDTAB (STRING)
C
C  START PROLOGUE
C
C  SUBROUTINE CKDTAB (STRING)
C  Replaces any tab character in a character string with one space.
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER STRING*(*), TAB*1
C
      TAB = CHAR(9)
   10 CONTINUE
      IND = INDEX(STRING,TAB)
      IF (IND .GT. 0) THEN
         STRING(IND:IND) = ' '
         GO TO 10
      ENDIF
C
C     end of SUBROUTINE CKDTAB
      RETURN
      END
C
C----------------------------------------------------------------------C
C                                                                      C
C*****precision > double
      DOUBLE PRECISION FUNCTION CKBSEC (NPTS, X, XX, TT)
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      REAL FUNCTION CKBSEC (NPTS, X, XX, TT)
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
C  START PROLOGUE
C  CKBSEC uses bisection to interpolate f(X), given X and other pairs
C  of X and f(X).
C
C  INPUT
C  NPTS	  - Integer scalar, total pairs of data.
C  X      - Real scalar, location for which f(X) is required.
C  XX(*)  - Real array, locations for which data is given.
C  TT(*)  - Real array, function values for locations given.
C
C  END PROLOGUE
C
      DIMENSION XX(NPTS), TT(NPTS)
C
      ZERO = 0.0
C
C     X is outside (1..NPTS)?
      IF (X .LE. XX(2)) THEN
         N = 2
         S = (TT(N) - TT(N-1)) / (XX(N) - XX(N-1))
      ELSEIF (X .GE. XX(NPTS-1)) THEN
         N = NPTS-1
         S = (TT(N+1) - TT(N)) / (XX(N+1) - XX(N))
      ELSE
         NLO = 1
         NHI = NPTS
         S   = ZERO
C
C        bisect interval
50       CONTINUE
         N = (NLO+NHI)/2
         IF (X .LT. XX(N)) THEN
            IF (X .LT. XX(N-1)) THEN
               NHI = N
               GO TO 50
            ELSEIF (X .EQ. XX(N-1)) THEN
               N = N-1
            ELSE
               S = (TT(N) - TT(N-1)) / (XX(N) - XX(N-1))
            ENDIF
         ELSEIF (X .GT. XX(N)) THEN
            IF (X .GT. XX(N+1)) THEN
               NLO = N
               GO TO 50
            ELSEIF (X .EQ. XX(N+1)) THEN
               N = N + 1
            ELSE
               S = (TT(N+1) - TT(N)) / (XX(N+1) - XX(N))
            ENDIF
         ENDIF
      ENDIF
C
c  100 CONTINUE
      CKBSEC      = TT(N) + S * (X - XX(N))
C
C     end of FUNCTION CKBSEC
      RETURN
      END
C                                                                      C
C                                                                      C
C----------------------------------------------------------------------C
C
      SUBROUTINE CKRDEX (I, RCKWRK, RD)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRDEX (I, RCKWRK, RD)*
C  Get/put the perturbation factor of the Ith reaction
C
C  INPUT
C  I         - Integer scalar, reaction index;
C              I > 0 gets RD(I) from RCKWRK
C              I < 0 puts RD(I) into RCKWRK
C  RCKWRK(*) - Real    workspace array; dimension at least LENRCK.
C
C  If I < 1:
C  RD        - Real scalar, perturbation factor for reaction I;
C              cgs units, mole-cm-sec-K.
C
C  OUTPUT
C  If I > 1:
C  RD        - Real scalar, perturbation factor for reaction I;
C              cgs units, mole-cm-sec-K.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
      PARAMETER (ONE = 1.0D0)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C      PARAMETER (ONE = 1.0)
C*****END precision > single
C
      INCLUDE 'ckstrt.h'
      DIMENSION RCKWRK(*)
C
      NI = NcCO + (IABS(I)-1)*(NPAR+1) + NPAR
      IF (I .GT. 0) THEN
         RD = RCKWRK(NI)
      ELSE
C
C          Assign the perturbation factor
C
         RCKWRK(NI) = RD
      ENDIF
C
C     end of SUBROUTINE CKRDEX
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKRHEX (K, RCKWRK, A6)
C
C  START PROLOGUE
C
C  SUBROUTINE CKRHEX (K, RCKWRK, A6)
C
C  Returns an array of the sixth thermodynamic polynomial
C  coefficients for a species, or changes their value,
C  depending on the sign of K.
C
C  INPUT
C  K         - Integer scalar, species index;
C              K>0 gets A6(*) from RCKWRK,
C              K<0 puts A6(*) into RCKWRK.
C  RCKWRK(*) - Real    workspace array; dimension at least LENRCK.
C
C  OUTPUT
C  A6(*)     - Real array, the 6th thermodynamic polynomial
C              coefficients for species K, over the number
C              of fit temperature ranges; dimension at least (MXTP-1),
C              where MXTP is the maximum number of temperatures used
C              to divide the thermodynamic fits.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      INCLUDE 'ckstrt.h'
      DIMENSION RCKWRK(*), A6(*)
C
      DO 100 L = 1, MXTP-1
         NA6 = NCAA + (L-1)*NCP2 + (IABS(K)-1)*NCP2T + NCP
         IF (K .GT. 0) THEN
            A6(L) = RCKWRK(NA6)
         ELSE
            RCKWRK(NA6) = A6(L)
         ENDIF
  100 CONTINUE
C
C     end of SUBROUTINE CKRHEX
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      CHARACTER*(*) FUNCTION CKCHUP(ISTR, ILEN)
      CHARACTER*(*) ISTR
      CHARACTER*1 LCASE(26), UCASE(26)
      DATA LCASE /'a','b','c','d','e','f','g','h','i','j','k','l','m',
     1            'n','o','p','q','r','s','t','u','v','w','x','y','z'/,
     2     UCASE /'A','B','C','D','E','F','G','H','I','J','K','L','M',
     3            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'/
C
      CKCHUP = ' '
      CKCHUP = ISTR(1:ILEN)
      JJ = MIN (LEN(CKCHUP), LEN(ISTR), ILEN)
      DO 10 J = 1, JJ
         DO 05 N = 1,26
            IF (ISTR(J:J) .EQ. LCASE(N)) CKCHUP(J:J) = UCASE(N)
   05    CONTINUE
   10 CONTINUE
C
C     end of FUNCTION CKCHUP
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKSAVE (LOUT, LSAVE, ICKWRK, RCKWRK, CCKWRK)
C
C  START PROLOGUE
C
C  SUBROUTINE CKSAVE (LOUT, LSAVE, ICKWRK, RCKWRK, CCKWRK)
C  Writes to a binary file information about a Chemkin linkfile,
C  pointers for the Chemkin Library, and Chemkin work arrays.
C
C  INPUT
C  LOUT      - Integer scalar, formatted output file unit number.
C  LSAVE     - Integer scalar, binary output file unit number.
C  ICKWRK(*) - Integer workspace array; dimension at least LENICK.
C  RCKWRK(*) - Real    workspace array; dimension at least LENRCK.
C  CCKWRK(*) - Character string workspace array;
C              dimension at least LENCCK.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
c      INCLUDE 'ckstrt.h'
      include 'ckcom.fh'
C
      DIMENSION ICKWRK(*), RCKWRK(*)
      CHARACTER*(*) CCKWRK(*)
      CHARACTER*16 FILVER, PRVERS, PREC
      LOGICAL KERR
C
      NPOINT = 85
      WRITE (LSAVE, ERR=999)
     *                FILVER, PREC, LENI, LENR, LENC,
C
C     include file for CHEMKIN-III cklib.f, dated: March 1, 1966
C
C     Integer constants
C
     1   NMM,  NKK,  NII,  MXSP, MXTB, MXTP, NCP,  NCP1, NCP2, NCP2T,
     2   NPAR, NLAR, NFAR, NLAN, NFAL, NREV, NTHB, NRLT, NWL,  NEIM,
     3   NJAN, NJAR, NFT1, NF1R, NEXC, NMOM, NXSM, NTDE, NRNU, NORD,
     4   MXORD, KEL, NKKI,
C
C     Integer pointers to character arrays in CCKWRK
C
     5   IcMM, IcKK,
C
C     Integer pointers to integer arrays in ICKWRK
C
     6   IcNC, IcPH, IcCH, IcNT, IcNU, IcNK, IcNS, IcNR, IcLT, IcRL,
     7   IcRV, IcWL, IcFL, IcFO, IcFT, IcKF, IcTB, IcKN, IcKT, IcEI,
     8   IcET, IcJN, IcF1, IcEX, IcMO, IcMK, IcXS, IcXI, IcXK, IcTD,
     9   IcTK, IcRNU,IcORD,IcKOR,IcKI, IcKTF,IcK1, IcK2,
C
C     Integer pointers to real variables and arrays in RCKWRK
C
     *   NcAW, NcWT, NcTT, NcAA, NcCO, NcRV, NcLT, NcRL, NcFL, NcKT,
     1   NcWL, NcJN, NcF1, NcEX, NcRU, NcRC, NcPA, NcKF, NcKR, NcRNU,
     2   NcKOR,NcK1, NcK2, NcK3, NcK4, NcI1, NcI2, NcI3, NcI4
C
C     END include file for cklib.f
C
C
      WRITE (LSAVE, ERR=999) (ICKWRK(L), L = 1, LENI)
      WRITE (LSAVE, ERR=999) (RCKWRK(L), L = 1, LENR)
      WRITE (LSAVE, ERR=999) (CCKWRK(L), L = 1, LENC)
      RETURN
C
  999 CONTINUE
      WRITE (LOUT,*)' Error writing Chemkin binary file information...'
      KERR = .TRUE.
C
C     end of SUBROUTINE CKSAVE
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C                                                                      C
      SUBROUTINE CKMXTP (ICKWRK, MAXTP)
C
C  START PROLOGUE
C
C  SUBROUTINE CKMXTP (ICKWRK, MAXTP)
C  Returns the maximum number of temperatures used in fitting the
C  thermodynamic properties of the species.
C
C  INPUT
C  ICKWRK(*) - Integer workspace array; dimension at least LENICK.
C
C  OUTPUT
C  MXTP      - Integer scalar, maximum number of temperatures used
C              to fit the thermodynamic properties of the species.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H, O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H, O-Z), INTEGER (I-N)
C*****END precision > single
C
c      INCLUDE 'ckstrt.h'
      DIMENSION ICKWRK(*)
C
      MAXTP = 1
C
C     end of SUBROUTINE CKMXTP
      RETURN
      END
C                                                                      C
C----------------------------------------------------------------------C
C
      INTEGER FUNCTION CKFRCH (STR)
C
C  START PROLGUE
C
C  INTEGER FUNCTION CKFRCH (STR)
C
C  Returns the index of the first non-blank, non-tab character in
C  a string.
C
C  INPUT
C  STR   - Character string
C
C  END PROLOGUE
C
      CHARACTER STR*(*), TAB*1
      INTEGER ILEN, I
C
      ILEN = LEN(STR)
      TAB  = CHAR(9)
      CKFRCH = 0
      DO 10 I = 1, ILEN
         IF (STR(I:I).EQ.' ' .OR. STR(I:I).EQ.TAB) GO TO 10
         CKFRCH = I
         RETURN
   10 CONTINUE
C
C     end of FUNCTION CKFRCH
      RETURN
      END
C                                                                      C
      INTEGER FUNCTION CKSLEN (LINE)
C
C  BEGIN PROLOGUE
C
C  INTEGER FUNCTION CKSLEN (LINE)
C  Returns the effective length of a character string, i.e.,
C  the index of the last character before an exclamation mark (!)
C  indicating a comment.
C
C  INPUT
C  LINE     - Character string.
C
C  OUTPUT
C  CKSLEN   - Integer scalar, the effective length of LINE.
C
C  END PROLOGUE
C
C*****precision > double
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)
C*****END precision > double
C*****precision > single
C      IMPLICIT REAL (A-H,O-Z), INTEGER (I-N)
C*****END precision > single
C
      CHARACTER LINE*(*)
      INTEGER CKLSCH, CKFRCH
      EXTERNAL CKLSCH, CKFRCH
C
      IND = CKFRCH(LINE)
      IF (IND.EQ.0 .OR. LINE(IND:IND).EQ.'!') THEN
         CKSLEN = 0
      ELSE
         IND = INDEX(LINE,'!')
         IF (IND .GT. 0) THEN
            CKSLEN = CKLSCH(LINE(1:IND-1))
         ELSE
            CKSLEN = CKLSCH(LINE)
         ENDIF
      ENDIF
C
C     end of FUNCTION CKSLEN
      RETURN
      END
C                                                                      C
