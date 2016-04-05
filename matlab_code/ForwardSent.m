function [Z, triples] = ForwardSent(depTree,params,oWe,Wsem)



if params.sus_depTree

    N = length(depTree.RNum); %sus: number of words
    Z = zeros(params.h, N); %sus: hxN. h is the size of the semantic space
    triples = zeros(3, N);
    n=1;

    for i=1:N  %sus: iterate over each word in the sentence

        % pc(2) increases monotonically thoughout the sentence
        pc = depTree.parentChild(i, :);  %sus: parent child: pc.
        if pc(1) <= 0 || pc(2) <=0, continue; end  %sus: when either parent or child are -1, there was no relationship found. Also ignore the root, ie. parent equal 0.
        if pc(1) > N || pc(2) > N, continue; end %sus: I'm not sure why this might happen, but ok.
        ix1 = depTree.sNums(pc(1));  %sus: id of parent word
        ix2 = depTree.sNums(pc(2));  %sus: id of child word
        if ix1==-1 || ix2==-1, continue; end

        ri = depTree.RNum(i); %sus: relation id
        w1 = oWe(:,ix1);  %sus: get vector of parent word
        w2 = oWe(:,ix2);  %sus: get vector of child word

        vcat = [w1; w2; 1]; % cat sus: concatenate weights of w1, w2 and 1 for bias
        vlin = Wsem * vcat; % mul sus: multiply
        v = params.f(vlin); % nonlin sus: apply nonlinearity
        Z(:,n) = v;

        triples(:,n) = [ix1,ix2,ri];  %sus: parent id, child id, relation id.
        n=n+1;
    end

    Z = Z(:, 1:n-1); % crop. sus: This is because some times the above loop is not fully
    % executed (e.g., pc(1)<=0). We don't need the extra columns full of zeros.
    % sus: n was increased 1 too many in the for loop, so here we have n-1.
    if nargout > 2
        triples = triples(:, 1:n-1); 
    end
   
% sus: change this code to a different structure to read the word vectors
else   
    N = length(depTree.RNum); %sus: number of words
    Z = zeros(params.h, N); %sus: hxN. h is the size of the semantic space
    triples = zeros(3, N);
    n=1;

    for i=1:N  %sus: iterate over each word in the sentence

        % pc(2) increases monotonically thoughout the sentence
        pc = depTree.parentChild(i, :);  %sus: parent child: pc.
        if pc(1) <= 0 || pc(2) <=0, continue; end  %sus: when either parent or child are -1, there was no relationship found. Also ignore the root, ie. parent equal 0.
        if pc(1) > N || pc(2) > N, continue; end %sus: I'm not sure why this might happen, but ok.
        ix1 = depTree.sNums(pc(1));  %sus: id of parent word
        ix2 = depTree.sNums(pc(2));  %sus: id of child word
        if ix1==-1 || ix2==-1, continue; end

        ri = depTree.RNum(i); %sus: relation id
        w1 = oWe(:,ix1);  %sus: get vector of parent word
        w2 = oWe(:,ix2);  %sus: get vector of child word

        vcat = [w1; w2; 1]; % cat sus: concatenate weights of w1, w2 and 1 for bias
        vlin = Wsem * vcat; % mul sus: multiply
        v = params.f(vlin); % nonlin sus: apply nonlinearity
        Z(:,n) = v;

        triples(:,n) = [ix1,ix2,ri];  %sus: parent id, child id, relation id.
        n=n+1;
    end

    Z = Z(:, 1:n-1); % crop. sus: This is because some times the above loop is not fully
    % executed (e.g., pc(1)<=0). We don't need the extra columns full of zeros.
    % sus: n was increased 1 too many in the for loop, so here we have n-1.
    if nargout > 2
        triples = triples(:, 1:n-1); 
    end
    
    
end

end
