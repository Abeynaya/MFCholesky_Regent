import "regent"

local c = regentlib.c

struct Config
{
  filename_matrix  : regentlib.string,
  filename_ord    : regentlib.string,
  filename_nbr	  : regentlib.string, 
  dimension       : int
}

local cstring = terralib.includec("string.h")

terra print_usage_and_abort()
  c.printf("Usage: regent mfND.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
  c.printf("  -m {file}     : Use {file} as matrix file.\n")
  c.printf("  -o {file}	    : Use {file} as ordering file\n")
  c.printf("  -n {file}	    : Use {file} as neighbors file\n")
  c.printf("  -d {value}	: Set the dimension \n")
  -- c.printf("  -o {file}     : Save the final edge to {file}. Will use 'edge.png' by default.\n")
  c.exit(0)
end

terra file_exists(filename : rawstring)
  var file = c.fopen(filename, "rb")
  if file == nil then return false end
  c.fclose(file)
  return true
end

terra Config:initialize_from_command()
  var filename_given = 0
  self.dimension = 3

  var args = c.legion_runtime_get_input_args()
  var i = 1
  var tot = args.argc
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
    elseif cstring.strcmp(args.argv[i], "-m") == 0 then
      i = i + 1
      if not file_exists(args.argv[i]) then
        c.printf("File '%s' doesn't exist!\n", args.argv[i])
        c.abort()
      end
      filename_given = filename_given+1
      self.filename_matrix = [regentlib.string](args.argv[i])
      
    elseif cstring.strcmp(args.argv[i], "-o") == 0 then
      i = i + 1
      if not file_exists(args.argv[i]) then
        c.printf("File '%s' doesn't exist!\n", args.argv[i])
        c.abort()
      end
      filename_given = filename_given+1
      self.filename_ord= [regentlib.string](args.argv[i])

    elseif cstring.strcmp(args.argv[i], "-n") == 0 then
      i = i + 1
      if not file_exists(args.argv[i]) then
        c.printf("File '%s' doesn't exist!\n", args.argv[i])
        c.abort()
      end
      filename_given = filename_given+1
      self.filename_nbr= [regentlib.string](args.argv[i])

      filename_given = filename_given+1
    elseif cstring.strcmp(args.argv[i], "-d") == 0 then
      i = i + 1
      self.dimension = c.atoi(args.argv[i])
    end
    i = i + 1
  end
  if filename_given<3 then
    c.printf("One of the input files missing\n\n")
    print_usage_and_abort()
  end

end
return Config

