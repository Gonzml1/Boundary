subroutine mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, M)
  implicit none
  integer, intent(in)    :: width, height, max_iter
  double precision, intent(in) :: xmin, xmax, ymin, ymax
  integer, intent(out)   :: M(height,width)
  integer :: i, j, n, iter
  double precision :: dx, dy, x, y, zx, zy, zx2, zy2

  dx = (xmax - xmin) / (width  - 1)
  dy = (ymax - ymin) / (height - 1)

  !$OMP PARALLEL DO PRIVATE(i,j,n,iter,x,y,zx,zy,zx2,zy2) SCHEDULE(dynamic)
  do i = 1, width
    x = xmin + (i-1) * dx
    do j = 1, height
      y = ymin + (j-1) * dy
      zx = 0.0d0
      zy = 0.0d0
      iter = 0
      do n = 1, max_iter
        zx2 = zx*zx - zy*zy + x
        zy2 = 2.0d0*zx*zy + y
        zx = zx2
        zy = zy2
        if (zx*zx + zy*zy > 4.0d0) exit
        iter = iter + 1
      end do
      M(j,i) = iter
    end do
  end do
  !$OMP END PARALLEL DO
end subroutine mandelbrot
