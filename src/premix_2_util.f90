subroutine open_premix_files( lin, lout, linmc, lrin, lrout, lrcvr, inputfile, &
    strlen, pathname, pathstrlen )

    integer, intent(in) :: lin, linmc, lrin, lrout, lrcvr,lout
    integer, intent(in) :: strlen, pathstrlen
    integer, intent(in) :: inputfile(strlen), pathname(pathstrlen)
    character(strlen) :: infile
    character(pathstrlen) :: path

    integer :: i


    do i=1,strlen
        infile(i:i) = ACHAR(inputfile(i))
    enddo
    do i=1,pathstrlen
        path(i:i) = ACHAR(pathname(i))
    enddo

    !write(*,*) 'inputfile name: ', trim(path)//trim(infile)
    !path = '../extras/premix_chemh/'
    OPEN(LIN,FORM='FORMATTED',STATUS='UNKNOWN',FILE=trim(path)//trim(infile))
    !OPEN(LOUT,FORM='FORMATTED',STATUS='UNKNOWN',FILE='./premix_log.out')
    OPEN(LINMC,FORM='FORMATTED',STATUS='UNKNOWN',FILE=trim(path)//'./tran.asc')
    OPEN(LRIN,FORM='UNFORMATTED',STATUS='UNKNOWN',FILE=trim(path)//'./rest.bin')
    OPEN(LROUT,FORM='UNFORMATTED',STATUS='UNKNOWN',FILE=trim(path)//'./save.bin')
    OPEN(LRCVR,FORM='UNFORMATTED',STATUS='UNKNOWN',FILE=trim(path)//'./recov.bin')

end
