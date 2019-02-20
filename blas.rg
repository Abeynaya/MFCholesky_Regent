
import "regent"

local linalg = {}

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cio = terralib.includec("stdio.h")
local std = terralib.includec("stdlib.h")


-- declare fortran-order 2D indexspace
local struct __f2d { y : int, x : int }
local f2d = regentlib.index_type(__f2d, "f2d")
linalg.f2d = f2d

local blas = terralib.includecstring [[
extern void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha,
                   double* A, int* lda, double* B, int* ldb, double* beta,
                   double* C, int* ldc);

extern void dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);

extern void dtrsm_(char* side, char *uplo, char* transa, char* diag,
                   int* m, int* n, double* alpha,
                   double *A, int *lda, double *B, int *ldb);

extern void dsyrk_(char *uplo, char* trans, int* n, int* k,
                   double* alpha, double *A, int *lda,
                   double* beta, double *C, int *ldc);

extern void dtrsv_(char *uplo, char* trans, char* diag,
                   int* n, double *A, int *lda, double *X, int *incx);

extern void dgemv_(char* trans, int* m, int* n, double* alpha,
                   double* A, int* lda, double* X, int* incx, double* beta,
                   double* Y, int* incy);             

]]

if os.execute("bash -c \"[ `uname` == 'Darwin' ]\"") == 0 then
  terralib.linklibrary("libblas.dylib")
  terralib.linklibrary("liblapack.dylib")
else
  terralib.linklibrary("libblas.so")
  terralib.linklibrary("liblapack.so")
end

function raw_ptr_factory(ty)
  local struct raw_ptr
  {
    ptr : &ty,
    offset : int,
  }
  return raw_ptr
end

local raw_ptr = raw_ptr_factory(double)

terra get_raw_ptr(xlo : int, ylo : int, xhi : int, yhi : int,
                  pr : c.legion_physical_region_t,
                  fld : c.legion_field_id_t)
  var fa = c.legion_physical_region_get_field_accessor_array_2d(pr, fld)
  var rect : c.legion_rect_2d_t
  var subrect : c.legion_rect_2d_t
  var offsets : c.legion_byte_offset_t[2]
  rect.lo.x[0] = ylo
  rect.lo.x[1] = xlo
  rect.hi.x[0] = yhi
  rect.hi.x[1] = xhi
  var ptr = c.legion_accessor_array_2d_raw_rect_ptr(fa, rect, &subrect, offsets)
  return raw_ptr { ptr = [&double](ptr), offset = offsets[1].offset / sizeof(double) }
end

-- Do factorization for SPD blocks
terra dpotrf_terra(xlo : int, ylo:int, xhi: int, yhi: int,
                   pr : c.legion_physical_region_t,
                   fld : c.legion_field_id_t)
  var uplo : rawstring = 'L'
  -- var n_ : int[1], bn_ : int[1]
  -- n_[0], bn_[0] = n, bn
  var bn_ : int[1] 
  bn_[0] = xhi-xlo +1
  var info : int[1]
  var rawA = get_raw_ptr(xlo, ylo, xhi, yhi, pr, fld)
  blas.dpotrf_(uplo, bn_, rawA.ptr, &(rawA.offset), info)
end


task dpotrf(rA : region(ispace(f2d), double))
where reads writes(rA)
do
	var bounds = rA.bounds
  var xlo = bounds.lo.x
  var ylo = bounds.lo.y
  var xhi = bounds.hi.x
  var yhi = bounds.hi.y
  dpotrf_terra(xlo,ylo,xhi,yhi, __physical(rA)[0], __fields(rA)[0])
end

-- Do triangular solves
terra dtrsm_terra(xloA : int, yloA:int, xhiA: int, yhiA: int,
                  xloB : int, yloB:int, xhiB: int, yhiB: int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)
  var side : rawstring = 'R'
  var uplo : rawstring = 'L'
  var transa : rawstring = 'T'
  var diag : rawstring = 'N'
  var rows_ : int[1], cols_ : int[1]
  rows_[0] =  yhiA-yloA+1
  cols_[0] =  xhiA-xloA+1
  var alpha : double[1] = array(1.0)
  var rawA = get_raw_ptr(xloA, yloA, xhiA, yhiA, prA, fldA)
  var rawB = get_raw_ptr(xloB, yloB, xhiB, yhiB, prB, fldB)
  blas.dtrsm_(side, uplo, transa, diag, rows_, cols_, alpha,
              rawB.ptr, &(rawB.offset), rawA.ptr, &(rawA.offset))
end

task dtrsm(rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads writes(rA), reads(rB)
do
  var boundsA = rA.bounds
  var boundsB = rB.bounds
  var xloA = boundsA.lo.x
  var yloA = boundsA.lo.y
  var xhiA = boundsA.hi.x
  var yhiA = boundsA.hi.y

  var xloB = boundsB.lo.x
  var yloB = boundsB.lo.y
  var xhiB = boundsB.hi.x
  var yhiB = boundsB.hi.y

  dtrsm_terra(xloA, yloA, xhiA, yhiA, xloB, yloB, xhiB, yhiB,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

-- GEMM for diagonal parts is matrix is symmetric
terra dsyrk_terra(xloA : int, yloA:int, xhiA: int, yhiA: int,
                  xloB : int, yloB:int, xhiB: int, yhiB: int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t)
  var uplo : rawstring = 'L'
  var trans : rawstring = 'N'
  var n_ : int[1], k_ : int[1]
  n_[0] = xhiA-xloA+1 --Size of matrix rA
  k_[0] = xhiB-xloB+1 --Number of columns of rB

  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)
  var rawA = get_raw_ptr(xloA, yloA, xhiA, yhiA, prA, fldA)
  var rawB = get_raw_ptr(xloB, yloB, xhiB, yhiB, prB, fldB)
  blas.dsyrk_(uplo, trans, n_, k_,
              alpha, rawB.ptr, &(rawB.offset),
              beta, rawA.ptr, &(rawA.offset))
end

task dsyrk(rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double))
where reads(rB),
      reduces -(rA)
do
  var boundsA = rA.bounds
  var boundsB = rB.bounds
  var xloA = boundsA.lo.x
  var yloA = boundsA.lo.y
  var xhiA = boundsA.hi.x
  var yhiA = boundsA.hi.y

  var xloB = boundsB.lo.x
  var yloB = boundsB.lo.y
  var xhiB = boundsB.hi.x
  var yhiB = boundsB.hi.y

  dsyrk_terra(xloA, yloA, xhiA, yhiA, xloB, yloB, xhiB, yhiB,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0])
end

-- General GEMM
terra dgemm_terra(xloA : int, yloA:int, xhiA: int, yhiA: int,
                  xloB : int, yloB:int, xhiB: int, yhiB: int,
                  xloC : int, yloC:int, xhiC: int, yhiC: int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t)
  var transa : rawstring = 'N'
  var transb : rawstring = 'T'
  var M_ : int[1], N_ : int[1], K_ : int[1]
  M_[0] = yhiA - yloA +1-- Rows of rB = rows of rA
  N_[0] = xhiA - xloA +1 -- Cols of rC^T = rows of rC = cols of rA
  K_[0] = xhiB - xloB +1 -- cols of rB = rows of rC^T = cols of rC

  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(xloA, yloA, xhiA, yhiA, prA, fldA)
  var rawB = get_raw_ptr(xloB, yloB, xhiB, yhiB, prB, fldB)
  var rawC = get_raw_ptr(xloC, yloC, xhiC, yhiC, prC, fldC)

  blas.dgemm_(transa, transb,M_, N_, K_,
              alpha, rawB.ptr, &(rawB.offset),
              rawC.ptr, &(rawC.offset),
              beta, rawA.ptr, &(rawA.offset))
end


task dgemm(rA : region(ispace(f2d), double),
           rB : region(ispace(f2d), double),
           rC : region(ispace(f2d), double))
where reduces -(rA), reads(rB, rC)
do
  var boundsA = rA.bounds
  var boundsB = rB.bounds
  var boundsC = rC.bounds

  var xloA = boundsA.lo.x
  var yloA = boundsA.lo.y
  var xhiA = boundsA.hi.x
  var yhiA = boundsA.hi.y

  var xloB = boundsB.lo.x
  var yloB = boundsB.lo.y
  var xhiB = boundsB.hi.x
  var yhiB = boundsB.hi.y

  var xloC = boundsC.lo.x
  var yloC = boundsC.lo.y
  var xhiC = boundsC.hi.x
  var yhiC = boundsC.hi.y

  dgemm_terra(xloA, yloA, xhiA, yhiA, xloB, yloB, xhiB, yhiB,
              xloC, yloC, xhiC, yhiC,
              __physical(rA)[0], __fields(rA)[0],
              __physical(rB)[0], __fields(rB)[0],
              __physical(rC)[0], __fields(rC)[0])
end

task factorize(rfront : region(ispace(f2d), double),
               sseps : int,
               snbrs : int)
where reads writes(rfront)
do
  var bounds = rfront.bounds
  var xlo = bounds.lo.x
  var ylo = bounds.lo.y
  var xhi = bounds.hi.x
  var yhi = bounds.hi.y

  -- potrf
  dpotrf_terra(xlo, ylo, xlo+sseps-1, ylo+sseps-1, 
         __physical(rfront)[0], __fields(rfront)[0])
  -- trsm 
  dtrsm_terra(xlo, ylo+sseps, xlo+sseps-1, yhi,
              xlo, ylo, xlo+sseps-1, ylo+sseps-1,
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rfront)[0], __fields(rfront)[0])
  -- syrk 
  dsyrk_terra(xlo+sseps, ylo+sseps, xhi, yhi,
              xlo, ylo+sseps, xlo+sseps-1, yhi,
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rfront)[0], __fields(rfront)[0])

end

task extend_add(rparent : region(ispace(f2d), double),
                par_idx : int,
                rchild : region(ispace(f2d), double),
                child_idx : int,
                rfrows : region(ispace(int2d), int))
where reads writes(rparent), reads(rchild, rfrows)
do

  -- Find the rows in the parent corresponding to the update
  var snbrs : int = rfrows[{x=child_idx, y=1}]
  var rind = region(ispace(int1d, snbrs),int)

  var l:int = 2
  var start = rfrows[{x=child_idx, y=0}]+2
  for i=start, start+snbrs, 1 do
    while(rfrows[{x=par_idx, y=l}] ~= rfrows[{x=child_idx,y=i}]) do
      l = l+1
    end
    rind[i-start]=l-2
    -- c.printf("print l = %d\n", rind[i-start])
  end

  var pbds = rparent.bounds
  var cbds = rchild.bounds

  for j = 0, snbrs, 1 do
    var fj = rind[j]
    for i=0, snbrs, 1 do
      var fi = rind[i]
      rparent[{y=pbds.lo.y+fj, x=pbds.lo.x+fi}] = rparent[{y=pbds.lo.y+fj, x=pbds.lo.x+fi}] 
                                                + rchild[{y=cbds.lo.y+j+start-2, x=cbds.lo.x+i+start-2}]
    end
  end

end

terra dtrsv_terra(xloA : int, yloA:int, xhiA: int, yhiA: int,
                  xloB : int, yloB:int, xhiB: int, yhiB: int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  code : int)
  -- var side : rawstring = 'R'
  var uplo : rawstring = 'L'
  var trans : rawstring = ''
  if code == 0 then
    trans ='N'
  else
    trans = 'T'
  end
  var diag : rawstring = 'N'
  var rows_ : int[1]
  rows_[0] =  yhiA-yloA+1
  -- var alpha : double[1] = array(1.0)

  var rawA = get_raw_ptr(xloA, yloA, xhiA, yhiA, prA, fldA)
  var rawB = get_raw_ptr(xloB, yloB, xhiB, yhiB, prB, fldB)

  blas.dtrsv_(uplo, trans, diag, rows_, 
              rawA.ptr, &(rawA.offset), rawB.ptr, &(rawB.offset))
end


terra dgemv_terra(xloA : int, yloA:int, xhiA: int, yhiA: int,
                  xloB : int, yloB:int, xhiB: int, yhiB: int,
                  xloC : int, yloC:int, xhiC: int, yhiC: int,
                  prA : c.legion_physical_region_t,
                  fldA : c.legion_field_id_t,
                  prB : c.legion_physical_region_t,
                  fldB : c.legion_field_id_t,
                  prC : c.legion_physical_region_t,
                  fldC : c.legion_field_id_t,
                  code : int)
  
  var trans : rawstring = ''
  if code == 0 then 
    trans ='N'
  else
    trans = 'T'
  end
 
  var M_ : int[1], N_ : int[1]
  M_[0] = yhiA - yloA +1--  rows of rA
  N_[0] = xhiA - xloA +1 -- cols of rA

  var alpha : double[1] = array(-1.0)
  var beta : double[1] = array(1.0)

  var rawA = get_raw_ptr(xloA, yloA, xhiA, yhiA, prA, fldA)
  var rawB = get_raw_ptr(xloB, yloB, xhiB, yhiB, prB, fldB)
  var rawC = get_raw_ptr(xloC, yloC, xhiC, yhiC, prC, fldC)

  blas.dgemv_(trans,M_, N_,
              alpha, rawA.ptr, &(rawA.offset),
              rawB.ptr, &(rawB.offset),
              beta, rawC.ptr, &(rawC.offset))
end

-- Forward solve
task fwd(rx : region(ispace(int2d), double),
          rfront : region(ispace(f2d), double),
          rfrows : region(ispace(int2d), int),
          rperm : region(ispace(int1d), int),
          front_idx : int,
          start : int)
where reads(rfront, rfrows, rperm), reads writes(rx)
do 
  var sseps : int = rfrows[{x=front_idx, y=0}]
  var snbrs : int = rfrows[{x=front_idx, y=1}]
  var rxn = region(ispace(int2d, {x=1, y=snbrs}), double)
  fill(rxn, 0.0)
  
  var bounds = rfront.bounds
  var xlo = bounds.lo.x
  var ylo = bounds.lo.y
  var xhi = bounds.hi.x
  var yhi = bounds.hi.y

  dtrsv_terra(xlo, ylo, xlo+sseps-1, ylo+sseps-1,
              0, start,0, start+sseps-1,
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rx)[0], __fields(rx)[0], 0)

  dgemv_terra(xlo, ylo+sseps, xlo+sseps-1, yhi, 
              0,start ,0,start+sseps-1,
              rxn.bounds.lo.x, rxn.bounds.lo.y, rxn.bounds.hi.x, rxn.bounds.hi.y, 
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rx)[0], __fields(rx)[0], 
              __physical(rxn)[0], __fields(rxn)[0],0)

-- some kind of extend add
var globid : int = start+sseps
for i=0, snbrs do
  while (rperm[globid]~= rfrows[{x=front_idx, y=sseps+2+i}]) do
    globid = globid+1
  end
  rx[{x=0,y=globid}] = rx[{x=0,y=globid}]+rxn[{x=0,y=i}]
end

end

-- Backward solve
task bwd(rx : region(ispace(int2d), double),
          rfront : region(ispace(f2d), double),
          rfrows : region(ispace(int2d), int),
          rperm : region(ispace(int1d), int),
          front_idx : int,
          start : int)
where reads(rfront, rfrows, rperm), reads writes(rx)
do 
  var sseps : int = rfrows[{x=front_idx, y=0}]
  var snbrs : int = rfrows[{x=front_idx, y=1}]
  var rxn = region(ispace(int2d, {x=1,y=snbrs}), double)
  fill(rxn, 0.0)

  -- Copy from x to xn 
  var globid : int = start +sseps
  for i=0, snbrs do
    while(rperm[globid] ~= rfrows[{x=rfrows, y=sseps+2+i}]) do
      globid = globid+1
    end
    rxn[{x=0,y=i}] = rx[{x=0,y=globid}]
  end
  
  var bounds = rfront.bounds
  var xlo = bounds.lo.x
  var ylo = bounds.lo.y
  var xhi = bounds.hi.x
  var yhi = bounds.hi.y

  dgemv_terra(xlo, ylo+sseps, xlo+sseps-1, yhi, 
              rxn.bounds.lo.x, rxn.bounds.lo.y, rxn.bounds.hi.x, rxn.bounds.hi.y, 
              0,start ,0,start+sseps-1,
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rxn)[0], __fields(rxn)[0],
              __physical(rx)[0], __fields(rx)[0], 1)

  dtrsv_terra(xlo, ylo, xlo+sseps-1, ylo+sseps-1,
              0, start,0, start+sseps-1,
              __physical(rfront)[0], __fields(rfront)[0],
              __physical(rx)[0], __fields(rx)[0], 1)
end

task verify(rrows : region(ispace(int1d), int),
             rcols : region(ispace(int1d), int),
             rvals : region(ispace(int1d), double),
             rb   : region(ispace(int2d), double),
             rx : region(ispace(int2d), double),
             rperm : region(ispace(int1d), int))
where reads(rrows, rcols, rvals, rperm, rx), reads writes(rb)
do 
var nvals = rrows.bounds.hi - rrows.bounds.lo
var nrows = rx.bounds.hi.y - rx.bounds.lo.y

for i=0, nvals do
  rb[{x=0,y=rrows[i]}] = rb[{x=0,y=rrows[i]}]-rvals[i]rx[{x=0,y=rcols[i]}]
  if rcols[i] ~= rrows[i] then
    rb[{x=0,y=rcols[i]}] = rb[{x=0,y=rcols[i]}]-rvals[i]rx[{x=0,y=rrows[i]}]
  end 
end

var sum : double = 0.0
for i=0, nrows do
  sum = sum + rb[{x=0,y=i}]*rb[{x=0,y=i}]
end

c.printf("||Ax-b|| = %8.4f", cmath.pow(sum, 0.5))

end


return linalg
-- task verify_result(n : int,
--                    org : region(ispace(f2d), double),
--                    res : region(ispace(f2d), double))
-- where reads(org, res)
-- do
--   c.printf("verifying results...\n")
--   for x = 0, n do
--     for y = x, n do
--       var v = org[f2d { x = x, y = y }]
--       var sum : double = 0
--       for k = 0, x + 1 do
--         sum += res[f2d { x = k, y = y }] * res[f2d { x = k, y = x }]
--       end
--       if cmath.fabs(sum - v) > 1e-6 then
--         c.printf("error at (%d, %d) : %.3f, %.3f\n", y, x, sum, v)
--       end
--     end
--   end
-- end