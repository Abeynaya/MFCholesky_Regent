import "regent"

local c = regentlib.c

struct Config
{
  filename_matrix  : int8[256],
  filename_ord    : int8[256],
  filename_nbr	  : int8[256]
  dimension       : int
}

local cstring = terralib.includec("string.h")

terra print_usage_and_abort()
  c.printf("Usage: regent mfND.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
  c.printf("  -m {file}     : Use {file} as matrix file.\n")
  c.printf("  -o {file}		: Use {file} as ordering file\n ")
  c.printf("  -n {file}		: Use {file} as neighbors file\n ")
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
  var filename_given = false

  -- cstring.strcpy(self.filename_edge, "edge.png")
  -- self.threshold = 80
  self.dimension = 3

  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
    elseif cstring.strcmp(args.argv[i], "-m") == 0 then
      i = i + 1
      if not file_exists(args.argv[i]) then
        c.printf("File '%s' doesn't exist!\n", args.argv[i])
        c.abort()
      end
      cstring.strcpy(self.filename_matrix, args.argv[i])
      
    elseif cstring.strcmp(args.argv[i], "-o") == 0 then
      i = i + 1
      cstring.strcpy(self.filename_ord, args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-n") == 0 then
      i = i + 1
      cstring.strcpy(self.filename_nbr, args.argv[i])
      filename_given = true
    elseif cstring.strcmp(args.argv[i], "-d") == 0 then
      i = i + 1
      self.dimension = c.atoi(args.argv[i])
    end
    i = i + 1
  end
  if not filename_given then
    c.printf("One of the input files missing\n\n")
    print_usage_and_abort()
  end
end


return Config

