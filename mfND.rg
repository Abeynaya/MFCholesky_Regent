import "regent"

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cio = terralib.includec("stdio.h")
local std = terralib.includec("stdlib.h")

-- Include blas and lapack solvers
local linalg = require("lin_alg")

-- declare fortran-order 2D indexspace
local f2d = linalg.f2d


terra skip_header(fp : &c.FILE)
  c.fscanf(fp, "%*[^\n]\n", nil)
end

terra read_line(f : &c.FILE, rcv : &double)
	var r : int,  cl:int, v : double
    c.fscanf(f, "%d %d %lg\n", &r, &cl, &v)
    rcv[0] = r
    rcv[1] = cl
    rcv[2] = v
end

task read_matrix(file : regentlib.string, 
				 rrows : region(ispace(int1d), int),
				 rcols : region(ispace(int1d), int),
				 rvals : region(ispace(int1d), double))
where writes(rrows, rcols, rvals)
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp)

	var rcv : double[3]
	read_line(fp, rcv)
	rrows[0]=[int](rcv[0])
	rcols[0]=[int](rcv[1])
	rvals[0]=[int](rcv[2])
	var nz = [int](rcv[2])

	for i=1, nz+1 do
		read_line(fp, rcv)
		rrows[i] = [int](rcv[0])-1
		rcols[i]=[int](rcv[1])-1
		rvals[i]=rcv[2]	
	end

	c.fclose(fp)

end

terra read_nz(file : regentlib.string)
	var m : int, n : int, nz : int

	var fp = c.fopen([rawstring](file), "rb")
	c.fscanf(fp, "%*[^\n]\n", nil)
	c.fscanf(fp, "%d %d %d\n", &m, &n, &nz)
	c.fclose(fp)
	return nz
end

terra read_char(fp : &c.FILE, 
				v : &int)
	c.fscanf(fp,"%d", &v[0])
end

task read_seps_region(rseps : region(ispace(int2d), int),
					  file 	: regentlib.string,
					  num_seps : int)
where writes(rseps)
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp)

	var v : int[0]
	for i=0, num_seps do
		read_char(fp, v)
		var size = v[0]
		rseps[{x=i, y=0}] = v[0]
		read_char(fp, v) -- skip

		for j=1, size+1 do
			read_char(fp, v)
			rseps[{x=i, y=j}]= v[0]
		end
	end

	c.fclose(fp)
end

terra get_levels(file : regentlib.string)
	var fp = c.fopen([rawstring](file), "rb")
	var nlvls : int = 0
	var nseps : int = 0 

	c.fscanf(fp, "%d %d\n", &nlvls, &nseps)
	c.fclose(fp) 
	return nlvls
end

task build_tree(nlvls : int,
				rtree : region(ispace(int2d),int))
where writes(rtree)
do
	var start : int = 0
	for i=nlvls, 0, -1 do
		var size : int = cmath.pow(2,i-1)
		for j=0, size do
			rtree[{x=i-1, y=j}]= start+j
		end
		start = start + size
	end
end

-- Add colored rectangle 
terra add_colored_rect(coloring : c.legion_domain_point_coloring_t,
                       color 	: int2d,
                       lo 		: int2d,
                       hi 		: int2d)
  var rect = c.legion_rect_2d_t{ lo = lo:to_point(), hi = hi:to_point()}
  c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(),
                                              c.legion_domain_from_rect_2d(rect))
end

-- Fill matrix 
task fill_matrix(rseps : region(ispace(int2d), int),
				 colx : int,
				 coly : int,
				 rrows : region(ispace(int1d), int),
				 rcols : region(ispace(int1d), int),
				 rvals : region(ispace(int1d), double),
				 submatrix   : region(ispace(f2d), double))
where writes(submatrix),
	  reads(rseps, rrows, rcols, rvals)
do
	var bounds = submatrix.bounds
	var xlo = bounds.lo.x
	var ylo = bounds.lo.y
	var xhi = bounds.hi.x
	var yhi = bounds.hi.y
	var nz = [int](rvals[0])

	var sep1_size = rseps[{x=colx, y=0}]
	var sep2_size = rseps[{x=coly, y=0}]

	var i : int = 0
	var j : int = 0
	
	var counter : int = 0
	for iter= 1, nz+1 do
		for i=1, sep1_size+1 do
			var idx_i = rseps[{x=colx, y=i}]
			if rrows[iter] == idx_i then
				for j=1, sep2_size+1 do
					var point1 : f2d = {y=j-1+ylo, x= i-1+xlo}
					var idx_j = rseps[{x=coly, y=j}]
					if rcols[iter] == idx_j then
						submatrix[point1] = rvals[iter]
						-- counter = counter +1
						break;
					end
				end

			elseif rcols[iter] == idx_i then
				for j=1, sep2_size+1 do
					var idx_j = rseps[{x=coly, y=j}]
					var point1 : f2d = {y=j-1+ylo, x= i-1+xlo}
					if  rrows[iter] == idx_j then
						submatrix[point1] = rvals[iter]
						-- counter = counter +1
						break;
					end
				end
			end
		end
	end

end

-- Verify results: LL^T = A
task verify_result(n : int,
                   org : region(ispace(f2d), double),
                   res : region(ispace(f2d), double))
where reads(org, res)
do
  c.printf("verifying results...\n")
  var check = true
  for x = 0, n do
    for y = x, n do
      var v = org[f2d { x = x, y = y }]
      var sum : double = 0
      for k = 0, x + 1 do
        sum += res[f2d { x = k, y = y }] * res[f2d { x = k, y = x }]
      end
      if cmath.fabs(sum - v) > 1e-6 then
        c.printf("error at (%d, %d) : %.3f, %.3f\n", y, x, sum, v)
        check = false
        break;
      end
    end
  end
  if check then
  	c.printf("VERIFIED: Cholesky decomposition successful\n")
  else 
  	c.printf(" Cholesky decomposition incorrect\n")
  end

end


task toplevel()
	-- Read in the sparse matrix stored in matrix market file
	var nrows : int, ncols : int, nz : int
	var matrix_file : regentlib.string = "lapl_30_3.mtx"
	-- var n :int = 20
	var d : int = 3

	nz 	  =  read_nz(matrix_file)
	var rrows = region(ispace(int1d, nz+1), int)
	var rcols = region(ispace(int1d, nz+1), int)
	var rvals = region(ispace(int1d, nz+1), double)

	read_matrix(matrix_file, rrows, rcols, rvals)

	nrows = rrows[0]
	ncols = rcols[0]

	c.printf("n_rows = %d, n_cols=%d, nz=%d\n",nrows,ncols,nz)

	-- Create logical region for the matrix 
	var r_org = region(ispace(f2d, {y=nrows,x=ncols}), double)
	var r_perm = region(ispace(f2d, {y=nrows,x=ncols}), double)

	-- -- Input file
	var filename : regentlib.string = "lapl_30_3_ord_9.txt"

	-- -- Get levels
	var nlvls : int = get_levels(filename)
	var num_seps : int = cmath.pow(2,nlvls)-1

	var N : int = [int](cmath.pow(nrows, [double](1.0/d)))
	
	var max_length : int = [int](2*cmath.pow(N, d-1)+1)
	var rseps = region(ispace(int2d, {x=num_seps, y= max_length}), int)
	-- Rread_seps(rseps, separators, max_length)
	read_seps_region(rseps, filename, num_seps)

	c.printf("SUCCESS: Read in the separators\n")

	
	-- Build tree 
	var rtree = region(ispace(int2d, {x= nlvls, y=cmath.pow(2,nlvls)}), int)
	build_tree(nlvls, rtree)

	c.printf("SUCCESS: Built tree of separators\n")

	-- Create the coloring
	var coloring = c.legion_domain_point_coloring_create() 
	--creates the coloring function. Think of this as a function remember the colors
	--we have used to color the different parts of our region

	var prev_size 		: int64 = 0
	var size 	  		: int64 = 0
	var sep_position = region(ispace(int1d, num_seps), int)


	__fence(__execution, __block)
    var ts_start = c.legion_get_current_time_in_micros()

	for lvl=nlvls-1, -1, -1 do
		var csize : int = cmath.pow(2, lvl) -- number of nodes at that level
		for j=0, csize, 1 do 
			var sep_idx : int = rtree[{x=lvl, y=j}]
			size = rseps[{x=sep_idx, y=0}] -- Size of the separator
			sep_position[sep_idx] = prev_size

			-- Create a 2D coloring according to seperator index
			var lo : int2d = {prev_size, prev_size}
			var hi : int2d = {prev_size+ size-1, prev_size+size-1}
			var color : int2d = {sep_idx, sep_idx}
			add_colored_rect(coloring, color, lo, hi)

			-- Check if the bounds make sense
			-- c.printf("color=(%d,%d), lo=(%d,%d), hi=(%d,%d)\n", color.x,color.y,lo.x,lo.y,hi.x,hi.y) 

			-- Go down the tree and assign colors for the children dependent on a seperator
			var k = lvl+1
			while k < nlvls do
				var nchild : int = cmath.pow(2,k-lvl)
				for p= 0, nchild, 1 do 
					-- var child_idx : int = tree:get(k):get(cmath.pow(2,k-lvl)*j+p)
					var child_idx : int = rtree[{x=k, y= cmath.pow(2,k-lvl)*j+p}]
					var width = rseps[{x=child_idx, y=0}]
					var start = sep_position[child_idx]

					-- Create a 2D coloring according to seperator index
					var clo : int2d = { prev_size, start }
					var chi : int2d = {prev_size + size-1, start + width -1}
					var ccolor : int2d = {child_idx, sep_idx}
					add_colored_rect(coloring, ccolor, clo, chi)
					-- c.printf("color=(%d,%d), lo=(%d,%d), hi=(%d,%d)\n", ccolor.x,ccolor.y,clo.x,
					-- 														clo.y,chi.x,chi.y) 
				end
				k = k+1
			end
			-- Update size
			prev_size = prev_size + size 
			-- end
		end
	end

	-- Create partition with our explicit colouring
	var pspace = ispace(int2d, {x = num_seps, y=num_seps}) 
	var pmatrix = partition(disjoint, r_perm, coloring, pspace)

	c.printf("SUCCESS: Partitioning done\n")

	-- Fill in the entries in our permuted matrix 
	for color in pspace do
		var colx = color.x
		var coly = color.y
		var submatrix = pmatrix[color]
		var bounds = submatrix.bounds
		var xlo = bounds.lo.x
		var ylo = bounds.lo.y
		var xhi = bounds.hi.x
		var yhi = bounds.hi.y
		
		if xlo <= xhi then	
			-- c.printf("color: {%d,%d}, bounds: lo:(%d %d), hi:(%d, %d)\n",colx, coly, xlo, ylo, xhi, yhi)
			fill_matrix(rseps, colx, coly, rrows, rcols, rvals, submatrix)
		end
	end

	--Print matrix entries
	-- for i=0, nrows do
	-- 	for j=0, ncols do
	-- 		var d : f2d = {y=i , x=j}
	-- 		if r_perm[d]==0.0 then
	-- 			c.printf("%3.1d",[int](r_perm[d]))
	-- 		else
	-- 			c.printf("%3.0f ", r_perm[d])
	-- 		end
	-- 	end
	-- 	c.printf("\n")
	-- end


	c.printf("SUCCESS: Matrix formed with ND ordering\n")


	-- -- Have a copy of the matrix to verify
	-- copy(r_perm, r_org)

	-- __fence(__execution, __block)
 --    var ts_start = c.legion_get_current_time_in_micros()

	for l = nlvls-1, -1, -1 do
		if l <= 4 then
			var nchild_at_l : int = cmath.pow(2, l)
			for i=0, nchild_at_l do
				var si : int = rtree[{x=l, y=i}]
				var rA = pmatrix[int2d{x=si, y=si}]
				dpotrf(rA) -- Do the factorization of diagonal blocks

				var c_idx : int = i
				-- Triangular solve
				for j=l-1, -1, -1 do -- go up the tree and find parents and grand parents
					var parent : int = [int](c_idx/2)
					var sepj : int = rtree[{x=j, y=parent}]
					var rB = pmatrix[int2d{x=si, y=sepj}]
					dtrsm(rB, pmatrix[int2d{x=si, y=si}]) -- reads writes to submatrix, reads pmatrix[]	
					c_idx = parent 
				end

			    c_idx = i
				for j=l-1, -1, -1 do
					var parent : int = [int](c_idx/2)
					var sepj : int = rtree[{x=j, y=parent}]
					var rB = pmatrix[int2d{x=si, y=sepj}]
					var rC = pmatrix[int2d{x=sepj, y=sepj}]
					dsyrk(rC,rB)

					var cp_parent = parent 

					for y= j-1, -1, -1 do 
						var grand_par : int = [int](cp_parent/2)
						var sepy : int = rtree[{x=y, y=grand_par}]
						var rD = pmatrix[ int2d{x=sepj, y=sepy} ]
						var rE = pmatrix[ int2d{x=si, y=sepy} ]

						dgemm(rD, rE, rB)
						cp_parent = grand_par 
					end
					c_idx = parent 
				end
			end

		else 
			var nchild_at_l : int = cmath.pow(2, l)
			for i=0, nchild_at_l do
				var si : int = rtree[{x=l, y=i}]

				var rA = pmatrix[int2d{x=si, y=si}]
				dpotrf_terra(rA.bounds.lo.x,rA.bounds.lo.y,
							 rA.bounds.hi.x,rA.bounds.hi.y, 
							 __physical(rA)[0], __fields(rA)[0]) -- Do the factorization of diagonal blocks

				var c_idx : int = i
				-- Triangular solve
				for j=l-1, -1, -1 do -- go up the tree and find parents and grand parents
					var parent : int = [int](c_idx/2)
					var sepj : int = rtree[{x=j, y=parent}]

					var rB = pmatrix[int2d{x=si, y=sepj}]
					dtrsm_terra(rB.bounds.lo.x,rB.bounds.lo.y,
							 	rB.bounds.hi.x,rB.bounds.hi.y, 
							 	rA.bounds.lo.x,rA.bounds.lo.y,
							 	rA.bounds.hi.x,rA.bounds.hi.y,
              					__physical(rB)[0], __fields(rB)[0],
             					__physical(rA)[0], __fields(rA)[0])	
					c_idx = parent 
				end

			    c_idx = i
				for j=l-1, -1, -1 do
					var parent : int = [int](c_idx/2)
					var sepj : int = rtree[{x=j, y=parent}]
					var rB = pmatrix[int2d{x=si, y=sepj}]
					var rC = pmatrix[int2d{x=sepj, y=sepj}]
					dsyrk_terra(rC.bounds.lo.x,rC.bounds.lo.y,
							 	rC.bounds.hi.x,rC.bounds.hi.y, 
							 	rB.bounds.lo.x,rB.bounds.lo.y,
							 	rB.bounds.hi.x,rB.bounds.hi.y,
              					__physical(rC)[0], __fields(rC)[0],
             				    __physical(rB)[0], __fields(rB)[0])

					var cp_parent = parent 

					for y= j-1, -1, -1 do 
						var grand_par : int = [int](cp_parent/2)
						var sepy : int = rtree[{x=y, y=grand_par}]
						var rD = pmatrix[ int2d{x=sepj, y=sepy} ]
						var rE = pmatrix[ int2d{x=si, y=sepy} ]

						dgemm_terra(rD.bounds.lo.x,rD.bounds.lo.y,
						 			rD.bounds.hi.x,rD.bounds.hi.y, 
						 			rE.bounds.lo.x,rE.bounds.lo.y,
						 			rE.bounds.hi.x,rE.bounds.hi.y,
						 			rB.bounds.lo.x,rB.bounds.lo.y,
						 			rB.bounds.hi.x,rB.bounds.hi.y,
          							__physical(rD)[0], __fields(rD)[0],
          							__physical(rE)[0], __fields(rE)[0],
          							__physical(rB)[0], __fields(rB)[0])
						cp_parent = grand_par 
					end
					c_idx = parent 
				end
			end
		end
	end


	c.printf("SUCCESS: Cholesky decomposition found\n")


	--Print matrix entries
	-- for i=0, nrows do
	-- 	for j=0, ncols do
	-- 		var d : f2d = {y=i , x=j}
	-- 		if r_perm[d]==0.0 then
	-- 			c.printf("%2.1d",[int](r_perm[d]))
	-- 		else
	-- 			c.printf("%8.4f ", r_perm[d])
	-- 		end
	-- 	end
	-- 	c.printf("\n")
	-- end


	__fence(__execution, __block)
	var ts_end = c.legion_get_current_time_in_micros()
  	c.printf("Total time: %.6f sec.\n", (ts_end - ts_start) * 1e-6)

  	-- Verify results
  	-- verify_result(nrows, r_org, r_perm)
   

end

regentlib.start(toplevel)