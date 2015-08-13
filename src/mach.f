      function d1mach(i) result(r)
        use iso_c_binding
        integer, intent(in) :: i
        double precision :: r
        interface
           function d1mach_c (ic) result(dc) bind(c)
             use, intrinsic :: iso_c_binding
             integer (kind=c_int), intent(in), value :: ic
             real (kind=c_double) :: dc
           end function d1mach_c
        end interface
        r = d1mach_c(i)
      end function d1mach

