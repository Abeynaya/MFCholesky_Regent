import "regent"

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cio = terralib.includec("stdio.h")
local std = terralib.includec("stdlib.h")

-- Include helper functions to read and write 
local helper = require("helper_fns")
-- Include blas and lapack solvers
local linalg = require("blas")
-- Include config file for command line arguments
local Config = require("nd_config")

-- declare fortran-order 2D indexspace
local f2d = linalg.f2d
local vec = linalg.vec

task read_ord(file 	: regentlib.string,
			  rperm : region(ispace(int1d),int),
			  rordptr : region(ispace(int1d),int),
			  num_seps : int)
where reads writes(rperm, rordptr) 
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp) -- Skip header

	var v : int[0]
	var size : int = 0
	var tot_size : int = 0

	for i=0, num_seps do
		read_char(fp, v) -- Skip
		read_char(fp, v) -- skip
		rordptr[i]=tot_size -- starting ptr of the separator i
		read_char(fp, v) 
		size = v[0]
		for j=0, size do
			read_char(fp, v)
			rperm[tot_size+j]=v[0]
		end
		tot_size = tot_size + size

	end
	rordptr[num_seps] = tot_size

	c.fclose(fp)
	return tot_size
end

task do_partition(num_seps: int,
				  rfrows : region(ispace(int2d),int),
				  rfronts : region(ispace(f2d), double),
				  pspace : ispace(int1d))
where reads(rfrows)
do
	var coloring = c.legion_domain_point_coloring_create() 
	var prev_size 		: int64 = 0
	var size 	  		: int64 = 0
	var max_size 		: int64 = 0

	-- leaves always comes first
	for si=0, num_seps do
		size = rfrows[{x=si, y=0}]+ rfrows[{x=si,y=1}]
		var lo : int2d = {prev_size,prev_size}
		var hi : int2d = {prev_size+size-1, prev_size+size-1}
		var color : int1d = si
		add_colored_rect(coloring, color, lo, hi)
		-- Update prev_size
		prev_size = prev_size + size

		-- Check if the bounds make sense
		-- c.printf("color=(%d), lo=(%d,%d), hi=(%d,%d)\n", color,lo.x,lo.y,hi.x,hi.y) 
	end
	
	var pfronts = partition(disjoint, rfronts, coloring, pspace)
	c.legion_domain_point_coloring_destroy(coloring)

	return pfronts
end


__demand(__inner)
task toplevel()
	var config : Config
	config:initialize_from_command()

	-- Read in the sparse matrix stored in matrix market file
	-- var nrows : int, ncols : int, nz : int
	var matrix_file  = config.filename_matrix
	
	var d = config.dimension
	var rcv : int[3]
	rcv	  =  read_nz(matrix_file)

	var nrows = rcv[0]
	var ncols = rcv[1]
	var nz = rcv[2]

	var rrows = region(ispace(int1d, nz), int)
	var rcolptrs = region(ispace(int1d, nz), int)
	var rvals = region(ispace(int1d, nz), double)
	read_matrix(matrix_file, rrows, rcolptrs, rvals)

	c.printf("n_rows = %d, n_cols=%d, nz=%d\n",nrows,ncols,nz)

	-- Ordering and neighbors file
	var ford = config.filename_ord
	var fnbr = config.filename_nbr
	var ftree = config.filename_tree

	-- Limits of rrows
	var N : int = [int](cmath.pow(nrows, [double](1.0/d)))
	var max_length : int = [int](2*cmath.pow(N, d-1)+1)

	-- Get levels and nseps
	var nls : int[2]
	nls = get_lvls_seps(ford)
	var nlvls = nls[0]
	var num_seps = nls[1]

	var tree : vec
	var lvls : vec
	tree:init(num_seps)
	lvls:init(nlvls)

	-- Read from file 
	for i=0, nlvls do
		lvls:set(i,0)
	end

	var fp = c.fopen([rawstring](ftree), "rb")
	skip_header(fp) -- Skip header
	var v : int[0]
	for i=0, num_seps do
		read_char(fp, v) -- level number
		lvls:add_one(v[0])
		read_char(fp, v) -- sep number
		read_char(fp, v) -- parent 
		tree:set(i,v[0])
	end
	c.fclose(fp)

	c.printf("SUCCESS: Read in the tree structure\n")

	-- Read in the separators
	var rfrows = region(ispace(int2d, {x=num_seps, y= 2*max_length}), int)
	

	var code : int = 0
	var sord = read_nodes_region(rfrows, ford, num_seps, code)
	c.printf("SUCCESS: Read in the separators\n")

	-- Read in the neighbors file
	code = 1
	var snbr = read_nodes_region(rfrows, fnbr, num_seps, code)
	c.printf("SUCCESS: Read in the neighbors\n")

	-- var rperm = region(ispace(int1d, nrows), int)
	-- var rordptr = region(ispace(int1d, num_seps+1),int)

	-- var sord = read_ord(rperm, rordptr, num_seps)

	-- var rnbrptr = region(ispace(int1d, num_seps+1), int)

    

	-- Partition region
	var rfronts = region(ispace(f2d, {y=sord+snbr, x = sord+snbr}), double)
    var pspace = ispace(int1d, num_seps)

    var pfronts = do_partition(num_seps, rfrows, rfronts, pspace)

	 __fence(__execution, __block)
     var ts_start = c.legion_get_current_time_in_micros()

	 c.printf("SUCCESS: Partitioning done\n")

	-- Form the fronts for each interface
	-- __demand(__parallel)
	-- for si=0, num_seps do
	-- 	-- c.printf("color=(%d), lo=(%d,%d), hi=(%d,%d)\n", si,front.bounds.lo.x,front.bounds.lo.y,front.bounds.hi.x,front.bounds.hi.y) 
	-- 	var rfront = pfronts[si]
	-- 	fill(rfront,0.0)
	-- end

	var si : int = 0
	for l=0, nlvls do
		var nseps_at_l : int = lvls:get(l)

		--__demand(__parallel)
		for s=0, nseps_at_l do
			fill_factorize(pfronts[si+s], rfrows, si+s, rrows, rcolptrs, rvals)
		end

		for s=0, nseps_at_l, 2 do
			var p =tree:get(si+s) -- parent
			if p~= -1 then
		 		var rparent = pfronts[p]
		 		extend_add(rparent, p, pfronts[si+s], si+s, rfrows)
			end
		end

		for s=1, nseps_at_l, 2 do
			var p =tree:get(si+s)-- parent
			if p~= -1 then
		 		var rparent = pfronts[p]
		 		extend_add(rparent, p, pfronts[si+s], si+s, rfrows)
			end
		end

		si = si+nseps_at_l
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
  		index = fwd(rx, pfronts[i], rfrows, rperm, i, index)
  	end

  	-- backward solve
  	for i=num_seps-1, -1, -1 do
 		-- var rxn = region(ispace(int2d, {x=1,y=snbrs}), double)
  		index = bwd(rx, pfronts[i], rfrows, rperm, i, index)
  	end

  	var rx_unperm = region(ispace(int2d, {x=1,y=nrows}), double)
  	unperm(rx_unperm, rperm, rx)
 

  	-- Verify Ax == b
  	verify(rrows, rcolptrs, rvals, rb, rx_unperm, rperm)

  	-- for i=0, nrows do
  	-- 	c.printf("%8.4f\n", rb[{x=0,y=i}])
  	-- end

end

regentlib.start(toplevel)
