function df_Wsem = BackwardSents(depTrees,params,oWe,Wsem,zs1,deltaParents)
% processes for all tress in same function call for more efficiency.

% init gradients
df_Wsem = zeros(size(Wsem));

numSentences = length(depTrees);
c = 1;
for s=1:numSentences
    depTree = depTrees{s};
    r = depTree.RStr;
    Z = zs1{s};
    
    N = length(depTree.RNum);
    n=1;
    for i=1:N
        pc = depTree.parentChild(i, :);
        if pc(1) <= 0 || pc(2) <=0, continue; end
        if pc(1) > N || pc(2) > N, continue; end
        ix1 = depTree.sNums(pc(1));
        ix2 = depTree.sNums(pc(2));
        if ix1==-1 || ix2==-1, continue; end
        
        % backprop non-linearity and chain
        dz = params.df(Z(:,n)) .* deltaParents(:,c);
        c=c+1;
        ri = depTree.RNum(i);
        
        % this was kid activation at forward prop time
        w1 = oWe(:,ix1);
        w2 = oWe(:,ix2);

        % backprop layer 1 matrix multiply
        vcat = [w1; w2; 1]; % cat
        df_Wsem = df_Wsem + dz * vcat';

        n=n+1;
    end

end

end
