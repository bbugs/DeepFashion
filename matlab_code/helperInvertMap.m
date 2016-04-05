function itos = helperInvertMap(stoi)
    imageix = unique(stoi);
    nix = length(imageix);
    itos = cell(nix, 1);
    for i=1:nix, itos{i} = find(stoi == i); end
end
