! Return saturation pressure of water at given temperature in Pa, T in C
! Uses data from G. van Wylen, R. Sonntag and C. Borgnakke, Fundamentals of
! Classical Thermodynamics, 4th ed. John Wiley & Sons, Inc., New York. 1993.
! ISBN 0-471-59395-8.
subroutine water_psat(Ta, p_saturation)

    implicit none

    real*8, intent(in) :: Ta
    real*8, intent(out) :: p_saturation

    integer, parameter :: ntable = 21
    real*8, dimension(ntable) :: Tsat, Psat

    integer :: i

    Tsat = (/0.01d0, 5.0d0, 10.0d0, 15.0d0, 20.0d0, 25.0d0, 30.0d0,&
             35.0d0, 40.0d0, 45.0d0, 50.0d0,  55.0d0, 60.0d0, 65.0d0,&
             70.0d0, 75.0d0, 80.0d0, 85.0d0, 90.0d0, 95.0d0, 100.0d0/)

    psat = (/0.6113d0, 0.8721d0, 1.2276d0, 1.7051d0, 2.3385d0, 3.1691d0, &
             4.2461d0, 5.6280d0, 7.3837d0, 9.5934d0, 12.350d0, 15.758d0, &
             19.941d0, 25.033d0, 31.188d0, 38.578d0, 47.390d0, 57.834d0, &
             70.139d0, 84.554d0, 101.325d0/)

    do i = 2, ntable
        if( Tsat(i) > Ta ) then
            exit
        end if
    enddo
    p_saturation = 1000.0d0*(psat(i-1) + (psat(i)-psat(i-1)) &
        * (Ta-Tsat(i-1))/(Tsat(i) - Tsat(i-1)))

    ! write(*,*) "Ta=", Ta
    ! write(*,*) "Tbracket=", Tsat(i-1), Tsat(i)
    ! write(*,*) "Pbracket=", psat(i-1), psat(i)
    ! write(*,*) "Psat=", p_saturation

end subroutine


! Use temperature dependency parameters 
! D0, K0, D1, K1, where D = D0 exp( D1/T ), etc to
! compute D and K, call other routine to compute 1D diffusion solution
! return values at list of ntimes
! On entry, flux must be at least ntimes long
subroutine wvtr_fun_global(times, flux, RH, Ta, thick_mil, &
                           D0, D1, K0, K1, ntimes)
    implicit none

    integer, intent(in) :: ntimes
    real*8, dimension(*), intent(in) :: times
    real*8, intent(in) :: RH, Ta, thick_mil, D0, D1, K0, K1
    real*8, dimension(*), intent(out) :: flux

    real*8 :: D, K
    D = D0 * exp(D1/(Ta+273.15d0))
    K = K0 * exp(K1/(Ta+273.15d0))
    
    call wvtr_fun(times, flux, RH, Ta, thick_mil, &
                  D, K, ntimes)

end subroutine wvtr_fun_global


! Compute flux based on 1D diffusion equation with given D, K,
! return values at list of ntimes
! On entry, flux must be at least ntimes long
subroutine wvtr_fun(times, flux, RH, Ta, thick_mil, D, K, ntimes)

    implicit none

    integer, intent(in) :: ntimes
    real*8, dimension(*), intent(in) :: times
    real*8, intent(in) :: RH, Ta, thick_mil, D, K
    real*8, dimension(*), intent(out) :: flux

    ! Number of terms in series solution
    integer, parameter :: nterms = 20 
    real*8, parameter :: Pa = 101.3325*1000.0d0

    ! Offset for instrument response
    real*8, parameter :: t0 = 80.0d0

    real*8 :: thickness
    real*8 :: activity, psat, prefac, series_sum
    real*8 :: pisq, invthicksq

    integer :: ii, itime

    pisq = 16*ATAN(1.0d0)
    thickness = thick_mil * 2.54d-5
    invthicksq = 1.0/(thickness*thickness)

    call water_psat(Ta, psat)
    activity = RH *  psat/Pa

    prefac = D*K*activity/thickness
    do itime = 1, ntimes
        if (times(itime) - t0 < 10) then
            flux(itime) = 0.0d0
        else
            flux(itime) = 1.0
            do ii = 1, nterms
                flux(itime) = flux(itime) + 2.0d0*(-1)**ii &
                    *exp( -D*ii*ii*pisq*(times(itime)-80.0d0)*invthicksq)
            end do
            flux(itime) = flux(itime)*prefac
        endif
    end do

end subroutine


! Wrapper to get data for experiments
! On input times, flux are empty arrays of length at least max_samples
! exp_number is experiment number 0..4
! RH, temperature, thickness, nsamples are scalars
! times, flux are vectors
! Actual data is in the include file spewed from python
! If nsamples comes back < 0 something is wrong
subroutine pmma_wvtr_data(exp_number, RH, temperature, thickness, sigma, &
                          times, flux, nsamples, max_samples)

    implicit none

    integer, intent(in) :: exp_number, max_samples
    real*8, intent(out) :: RH, temperature, thickness, sigma
    real*8, intent(out), dimension(*) :: times, flux
    integer, intent(out) :: nsamples

    integer, parameter :: max_expt = 5

    real*8, dimension(max_expt) :: rh_list, temp_list, thickness_list, &
                                   sigma_list
    real*8 :: psat

    include 'ftn_data.f90'

    if( ndata(exp_number) > max_samples ) then
        write(*,*) "No space for samples, need ", ndata(exp_number)
        nsamples = -1
        return
    end if

    nsamples = ndata(exp_number)
    RH = rh_list(exp_number)
    temperature = temp_list(exp_number)
    thickness = thickness_list(exp_number)

    ! Return sigma = 10sigma - include some model error, other sources of error
    sigma = sigma_list(exp_number)*10.0d0

    select case (exp_number)
      case (1)
        times(1:ndata(1)) = ts_1
        flux(1:ndata(1)) = js_1
      case (2)
        times(1:ndata(2)) = ts_2
        flux(1:ndata(2)) = js_2
      case (3)
        times(1:ndata(3)) = ts_3
        flux(1:ndata(3)) = js_3
      case (4)
        times(1:ndata(2)) = ts_4
        flux(1:ndata(2)) = js_4
      case default
          write(*,*) 'No data available for experiment', exp_number
    end select

end subroutine pmma_wvtr_data

! Compute ln likelihood for a given set of parameter values
subroutine get_prob(D0, D1, K0, K1, L)

    implicit none

    real*8, intent(in) :: D0, D1, K0, K1
    real*8, intent(out) :: L

    real*8 :: RH, Ta, thickness, sigma
    integer, parameter :: max_data_size = 500
    real*8, dimension(max_data_size) :: times, flux_compute, flux_data
    integer :: N
    real*8 inv_sigma2

    integer :: iexp, ii
    integer :: nexp = 2

    ! Data part
    L = 0.0d0
    do iexp = 1, nexp
        call pmma_wvtr_data (iexp, RH, Ta, thickness, sigma, &
                             times, flux_data, N, max_data_size)
        inv_sigma2 = 1.0d0/(sigma*sigma)
        call wvtr_fun_global(times, flux_compute, RH, Ta, thickness, &
                             D0, D1, K0, K1, N)
        L = L - 0.5d0 * sum( (flux_compute(1:N) - flux_data(1:N))**2 * inv_sigma2 )
    end do

end subroutine get_prob


! Driver / test 
program main

    implicit none

    integer, parameter :: max_data_size = 500
    real*8, dimension(max_data_size) :: ts, js, jsfcn, jsfcn2
    real*8 :: rel_humidity, T, thickness, sigma
    integer :: N, i

    real*8 D, K
    real*8 D0, D1, K0, K1
    real*8 L
    
    ! Get an experiment, evaluate the transport function with reasonable
    ! parameters
    call pmma_wvtr_data (1, rel_humidity, T, thickness, sigma, &
                         ts, js, N, max_data_size)
    D = 3.44621430d-12   
    K = 3.08151517d+02


    call wvtr_fun( ts, jsfcn, rel_humidity, T, thickness, D, K, N)


    D0 = 3.7627936832728142d-07
    K0 = 0.0052442859717903682d0
    D1 = -3631.6338277973091d0
    K1 =  3437.6670962988424d0
    call wvtr_fun_global(ts, jsfcn2, rel_humidity, T, thickness, &
                         D0, D1, K0, K1, N)
    !
    
    call get_prob(D0, D1, K0, K1, L)
    write(*,*) "L = ", L
    do i = 1, N
        write(*,*) ts(i), '  ', js(i), jsfcn(i), jsfcn2(i), &
                   js(i) + sigma, js(i)-sigma
    enddo

end program
