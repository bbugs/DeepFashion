function val = getparam(p, fname, defval)

if isfield(p, fname)
    val = p.(fname);
else
    val = defval;
end

end
