import "regent"

local helper = {}

-- C APIs
local c = regentlib.c
local cmath = terralib.includec("math.h")
local cio = terralib.includec("stdio.h")
local std = terralib.includec("stdlib.h")

local linalg = require("blas")
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
				 rcolptrs : region(ispace(int1d), int),
				 rvals : region(ispace(int1d), double))
where writes(rrows, rcolptrs, rvals)
do
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp)

	var rcv : double[3]
	read_line(fp, rcv) -- skip

	var nz = [int](rcv[2])

	var cptr : int = 1

	for i=0, nz do
		read_line(fp, rcv)
		rrows[i] = [int](rcv[0])-1
		rvals[i]=rcv[2]	

		if ([int](rcv[1]) ~= cptr-1) then
			rcolptrs[cptr]=i
			cptr = cptr+1
		end

	end
	rcolptrs[cptr]=nz

	c.fclose(fp)

end

terra read_nz(file : regentlib.string)
	var rcv : int[3]
	var m : int, n : int, nz : int
	var fp = c.fopen([rawstring](file), "rb")
	c.fscanf(fp, "%*[^\n]\n", nil)
	c.fscanf(fp, "%d %d %d\n", &m, &n, &nz)
	c.fclose(fp)
	rcv[0] = m 
	rcv[1] = n 
	rcv[2] = nz 

	return rcv
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
	var tot_size : int = 0
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp) -- Skip header

	var v : int[0]
	for i=0, num_seps do
		read_char(fp, v) -- Skip
		read_char(fp, v) -- skip

		read_char(fp, v) 

		if code == 0 then
			var size = v[0]
			tot_size = tot_size + size
			rfrows[{x=i, y=0}] = size

			for j=2, size+2 do
				read_char(fp, v)
				rfrows[{x=i, y=j}]= v[0]
			end
		else
			var size = v[0]
			tot_size = tot_size + size

			rfrows[{x=i, y=1}] = size
			var start = rfrows[{x=i,y=0}]+2

			for j=start, start+size do
				read_char(fp, v)
				rfrows[{x=i, y=j}]= v[0]
			end
		end

	end

	c.fclose(fp)
	return tot_size
end

task read_tree(file : regentlib.string,
			   rtree : region(ispace(int1d),int),
			   rlvls : region(ispace(int1d),int),
			   nseps : int)
where writes(rtree), reads writes(rlvls)
do
	fill(rlvls,0)
	var fp = c.fopen([rawstring](file), "rb")
	skip_header(fp) -- Skip header
	
	var v : int[0]

	for i=0, nseps do
		read_char(fp, v) -- level number
		rlvls[v[0]]=rlvls[v[0]]+1

		read_char(fp, v) -- sep number
		read_char(fp, v) -- parent 
		rtree[i]=v[0]		
	end
end

task unperm(rx_unperm : region(ispace(int2d),double),
			rperm :region(ispace(int1d),int),
			rx : region(ispace(int2d),double) )
where reads writes(rx_unperm), reads(rperm, rx)
do
	var nrows = rx.bounds.hi.y - rx.bounds.lo.y + 1
 	for i=0, nrows do
  		rx_unperm[{x=0,y=rperm[i]}] = rx[{x=0,y=i}]
  	end
end

terra get_lvls_seps(file : regentlib.string)
	var fp = c.fopen([rawstring](file), "rb")
	var nlvls : int = 0
	var nseps : int = 0 
	var nls : int[2]

	c.fscanf(fp, "%d %d\n", &nlvls, &nseps)
	c.fclose(fp) 
	nls[0] = nlvls
	nls[1] = nseps
	return nls
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

	var csize = rfrows[{x=si, y=0}] 
	var nsize = rfrows[{x=si, y=1}]

	-- Ass part 
	for i=0, csize do
		var ci = rfrows[{x=si,y=2+i}]
		var cptr = rcols[ci+1] -- start index of that column ci  
		for j=0, csize do
			for l=cptr, rcols[ci+2] do
				if(rfrows[{x=si, y=j+2}]== rrows[l]) then 
					submatrix[{y= ylo+ j,x=xlo+i }]= rvals[l]
					submatrix[{y= ylo+ i,x=xlo+j }]= rvals[l]
					break
				elseif (rfrows[{x=si, y=j+2}]<rrows[l]) then
					break
				end
			end

		end
	end

	-- Ans part
	var m :int = 0
	for i=0, csize do 
		var ci = rfrows[{x=si,y=2+i}]
		for j=0, nsize do
			var ri = rfrows[{x=si, y=j+2+csize}]

			if ci<ri then
				var cptr = rcols[ci+1]
				for l=cptr, rcols[ci+2] do
					if(rfrows[{x=si, y=j+2+csize}]==rrows[l]) then
						submatrix[{y=ylo+j+csize, x=xlo+i}]=rvals[l]
						break
					elseif (rfrows[{x=si, y=j+2+csize}]<rrows[l]) then
						break
					end
				end
			else 
				var cptr = rcols[ri+1]
				for l=cptr, rcols[ri+2] do
					if(rfrows[{x=si, y=i+2}]==rrows[l]) then
						submatrix[{y=ylo+j+csize, x=xlo+i}]=rvals[l]
						break
					elseif (rfrows[{x=si, y=i+2}]<rrows[l]) then
						break
					end
				end
			end
		end
	end

end

task form_permutation(rperm : region(ispace(int1d),int),
					  rfrows : region(ispace(int2d),int),
					  num_seps : int)
where writes(rperm), reads(rfrows) 
do 
	var tot_ord_size : int = 0
	-- Add to perm vectors
	for si=0, num_seps do
		var ord_size : int = rfrows[{x=si, y=0}]
		for j=0, ord_size do
			rperm[j+tot_ord_size] = rfrows[{x=si, y=j+2}]
		end
		tot_ord_size = tot_ord_size + ord_size
	end
end



return helper