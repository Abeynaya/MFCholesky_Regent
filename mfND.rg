import "regent"

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cio = terralib.includec("stdio.h")
local std = terralib.includec("stdlib.h")

-- Include blas and lapack solvers
local linalg = require("blas")
-- Include config file for command line arguments
local Config = require("nd_config")

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

task read_nodes_region(rfrows : region(ispace(int2d), int),
					  	file 	: regentlib.string,
					  	num_seps : int,
					  	code : int)
where reads writes(rfrows)
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp) -- Skip header

	var v : int[0]
	for i=0, num_seps do
		read_char(fp, v) -- Skip
		read_char(fp, v) -- skip

		read_char(fp, v) 

		if code == 0 then
			var size = v[0]
			rfrows[{x=i, y=0}] = size

			for j=2, size+2 do
				read_char(fp, v)
				rfrows[{x=i, y=j}]= v[0]
			end
		else
			var size = v[0]
			rfrows[{x=i, y=1}] = size
			var start = rfrows[{x=i,y=0}]+2

			for j=start, start+size do
				read_char(fp, v)
				rfrows[{x=i, y=j}]= v[0]
			end
		end

	end

	c.fclose(fp)
end

task read_tree(file : regentlib.string,
			   rtree : region(ispace(int1d),int),
			   rlvls : region(ispace(int1d),int),
			   nseps : int)
where writes(rtree, rlvls)
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp) -- Skip header
	
	for i=0, nseps do
		read_char(fp, v) -- level number
		rlvls[v[0]]=rlvls[v[0]]+1

		read_char(fp, v) -- sep number
		read_char(fp, v) -- parent 
		rtree[i]=v[0]		
	end
end

terra get_nseps(file : regentlib.string)
	var fp = c.fopen([rawstring](file), "rb")
	var nlvls : int = 0
	var nseps : int = 0 

	c.fscanf(fp, "%d %d\n", &nlvls, &nseps)
	-- c.fscanf(fp, "%d\n", &nseps)
	c.fclose(fp) 
	return nlvls, nseps
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
                       color 	: int1d,
                       lo 		: int2d,
                       hi 		: int2d)
  var rect = c.legion_rect_2d_t{ lo = lo:to_point(), hi = hi:to_point()}
  c.legion_domain_point_coloring_color_domain(coloring, color:to_domain_point(),
                                              c.legion_domain_from_rect_2d(rect))
end

-- Fill matrix 
task fill_matrix(rfrows : region(ispace(int2d), int),
				 si : int,
				 rrows : region(ispace(int1d), int),
				 rcols : region(ispace(int1d), int),
				 rvals : region(ispace(int1d), double),
				 submatrix   : region(ispace(f2d), double))

where writes(submatrix),
	  reads(rfrows, rrows, rcols, rvals)
do
	var bounds = submatrix.bounds
	var xlo = bounds.lo.x
	var ylo = bounds.lo.y
	var xhi = bounds.hi.x
	var yhi = bounds.hi.y
	var nz = [int](rvals[0])

	var sep1_size = rfrows[{x=si, y=0}] 
	var sep2_size = sep1_size + rfrows[{x=si, y=1}]

	-- var rsubcol = rseps[{x=si}] | rnbrs[{x=si}]

	for iter= 1, nz+1 do
		for i=2, sep1_size+2 do
			var idx_i = rfrows[{x=si, y=i}]
			if rrows[iter] == idx_i then
				for j=2, sep2_size+2 do
					var point1 : f2d = {y=j-2+ylo, x= i-2+xlo}

					var idx_j = rfrows[{x=si, y=j}]

					if rcols[iter] == idx_j  then
						submatrix[point1] = rvals[iter]
						-- counter = counter +1
						break;
					end
				end

			elseif rcols[iter] == idx_i then
				for j=2, sep2_size+2 do
					var point1 : f2d = {y=j-2+ylo, x= i-2+xlo}

					var idx_j = rfrows[{x=si, y=j}]

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

task toplevel()
	var config : Config
	config:initialize_from_command()

	-- Read in the sparse matrix stored in matrix market file
	var nrows : int, ncols : int, nz : int
	var matrix_file  = config.filename_matrix
	
	var d = config.dimension
	nz 	  =  read_nz(matrix_file)
	var rrows = region(ispace(int1d, nz+1), int)
	var rcols = region(ispace(int1d, nz+1), int)
	var rvals = region(ispace(int1d, nz+1), double)

	read_matrix(matrix_file, rrows, rcols, rvals)

	nrows = rrows[0]
	ncols = rcols[0]

	c.printf("n_rows = %d, n_cols=%d, nz=%d\n",nrows,ncols,nz)

	-- Ordering and neighbors file
	var ord = config.filename_ord
	var nbr = config.filename_nbr
	var tree = config.filename_tree

	-- Limits of rrows
	-- var max_length : int = nrows 
	var N : int = [int](cmath.pow(nrows, [double](1.0/d)))
	var max_length : int = [int](2*cmath.pow(N, d-1)+1)

	-- Get levels
	--var nlvls : int = get_levels(ord)
	-- var num_seps : int = get_nseps(ord)
	var nlvs : int, num_seps : int = get_nseps(ord)

	-- Read in the separators
	var rfrows = region(ispace(int2d, {x=num_seps, y= 2*max_length}), int)

	var code : int = 0
	read_nodes_region(rfrows, ord, num_seps, code)
	c.printf("SUCCESS: Read in the separators\n")

	-- Read in the neighbors file
	code = 1
	read_nodes_region(rfrows, nbr, num_seps, code)
	c.printf("SUCCESS: Read in the neighbors\n")
	
	-- Build tree 
	-- var rtree = region(ispace(int2d, {x= nlvls, y=cmath.pow(2,nlvls)}), int)
	-- build_tree(nlvls, rtree)
	-- c.printf("SUCCESS: Built tree of separators\n")
	var rtree = region(ispace(int1d, num_seps), int)
	var rlvls = region(ispace(int1d, nlvls), int)
	fill(rlvls,0)
	read_tree(tree,rtree, rlvls, num_seps)
	c.printf("SUCCESS: Read in the tree structure\n")


	-- permutation vector
	var rperm = region(ispace(int1d, nrows), int)
	var tot_ord_size : int = 0
	-- Add to perm vectors
	for si=0, num_seps do
		var ord_size : int = rfrows[{x=si, y=0}]
		for j=0, ord_size do
			rperm[j+tot_ord_size] = rfrows[{x=si, y=j+2}]
		end
		tot_ord_size = tot_ord_size + ord_size
	end


	-- Create a 2D coloring for different fronts
	var coloring = c.legion_domain_point_coloring_create() 
	--creates the coloring function. Think of this as a function remember the colors
	--we have used to color the different parts of our region

	var prev_size 		: int64 = 0
	var size 	  		: int64 = 0
	-- var sep_position = region(ispace(int1d, num_seps), int)
	var max_size 		: int64 = 0

	-- leaves always comes first
	for si=0, num_seps do
		size = rfrows[{x=si, y=0}]+ rfrows[{x=si,y=1}]
		-- var lo : int2d = {prev_size,prev_size}
		-- var hi : int2d = {prev_size+size-1, prev_size+size-1}
		-- var color : int2d = {si,si}
		var lo : int2d = {0,prev_size}
		var hi : int2d = {size-1,prev_size+size-1}
		var color : int1d = si
		
		add_colored_rect(coloring, color, lo, hi)
		-- Update prev_size
		prev_size = prev_size + size

		if size > max_size then
			max_size = size 
		end
		-- Check if the bounds make sense
		-- c.printf("color=(%d), lo=(%d,%d), hi=(%d,%d)\n", color,lo.x,lo.y,hi.x,hi.y) 
	end

	-- Create the region of fronts
	-- var rfronts_size = prev_size
	-- var rfronts = region(ispace(f2d, {y=prev_size, x = prev_size}), double)
	var rfronts_size = max_size
	var rfronts = region(ispace(f2d, {y=max_size, x = prev_size}), double)

	-- Create the partition 
	-- var pspace = ispace(int2d, {x=num_seps, y=num_seps})
	var pspace = ispace(int1d, num_seps)
	var pfronts = partition(disjoint, rfronts, coloring, pspace)

	c.printf("SUCCESS: Partitioning done\n")


	__fence(__execution, __block)
    var ts_start = c.legion_get_current_time_in_micros()

	-- Form the fronts for each interface
	for si=0, num_seps do
		--c.printf("si = %d",si)
		var front = pfronts[si]
		-- c.printf("color=(%d), lo=(%d,%d), hi=(%d,%d)\n", si,front.bounds.lo.x,front.bounds.lo.y,front.bounds.hi.x,front.bounds.hi.y) 

		fill_matrix(rfrows, si, rrows, rcols, rvals, front)
	end

	-- Print fronts
	-- for si=0, num_seps do
	-- 	var bds = pfronts[{x=si, y=si}].bounds 
	-- 	var nr = bds.hi.y - bds.lo.x +1
	-- 	var nc = bds.hi.x - bds.lo.x +1
	-- 	for i=0, nr do
	-- 		for j=0, nc do
	-- 			var d : f2d = {y=bds.lo.y+i , x=bds.lo.x+j}
	-- 			if rfronts[d]==0.0 then
	-- 				c.printf("%2.1d",[int](rfronts[d]))
	-- 			else
	-- 				c.printf("%3.0f ", rfronts[d])
	-- 			end	
	-- 		end
	-- 		c.printf("\n")
	-- 	end
	-- 	c.printf("\n \n ")
	-- end

	for si=0, num_seps do
		var rchild = pfronts[si]
		factorize(rchild, rfrows[{x=si, y=0}], rfrows[{x=si, y=1}])

		var p =rtree[si]
		-- Extend add to the parent
		if p~= -1 then
		 	var rparent = pfronts[p]
		 	extend_add(rparent, p, rchild, si, rfrows)
		end
	end

	__fence(__execution, __block)
	var ts_end = c.legion_get_current_time_in_micros()
	c.printf("SUCCESS: Factorization done\n")
  	c.printf("Total time: %.6f sec.\n", (ts_end - ts_start) * 1e-6)

  	-- Solve 
  	var rx = region(ispace(int2d, {x=1,y=nrows}),double)
  	var rb = region(ispace(int2d, {x=1,y=nrows}),double)
  	fill(rx, 2.0)
  	copy(rx, rb)

  	-- Forward solve
  	var index : int = 0
  	for i=0, num_seps do
  		fwd(rx, pfronts[i], rfrows, rperm, i, index)
  		index = index+rfrows[{x=i, y=0}]
  	end

  	-- backward solve
  	for i=num_seps-1, -1, -1 do
  		bwd(rx, pfronts[i], rfrows, rperm, i, index)
  		index = index-rfrows[{x=i, y=0}]
  	end

  	var rx_unperm = region(ispace(int2d, {x=1,y=nrows}), double)
  	for i=0, nrows do
  		rx_unperm[{x=0,y=rperm[i]}] = rx[{x=0,y=i}]
  	end

  	-- Verify Ax == b
  	verify(rrows, rcols, rvals, rb, rx_unperm, rperm)

  	-- for i=0, nrows do
  	-- 	c.printf("%8.4f\n", rb[{x=0,y=i}])
  	-- end


	-- -- Print fronts
	-- for si=num_seps-1, num_seps do
	-- 	var bds = pfronts[{x=si, y=si}].bounds 
	-- 	var nr = bds.hi.y - bds.lo.x +1
	-- 	var nc = bds.hi.x - bds.lo.x +1
	-- 	for i=0, nr do
	-- 		for j=0, nc do
	-- 			var d : f2d = {y=bds.lo.y+i , x=bds.lo.x+j}
	-- 			c.printf("%8.4f ", rfronts[d])	
	-- 		end
	-- 		c.printf("\n")
	-- 	end
	-- 	c.printf("\n \n")
	-- end

end

regentlib.start(toplevel)
