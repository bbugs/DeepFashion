
% sets params.f and params.df based on params.actFunc
% sus: this file assumes that params is already in the workspace and
% params.actFunc has been defined
switch params.actFunc
    case 'linthresh',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f     = @linthresh;
        params.wordf = @linthresh;
        params.df    = @linthreshg;
        params.worddf= @linthreshg;
    case 'linthresh1',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f     = @linthresh1;
        params.wordf = @linthresh1;
        params.df    = @linthreshg1;
        params.worddf= @linthreshg1;
    case 'identity',
        params.f     = @identity;
        params.wordf = @identity;
        params.df    = @identityg;
        params.worddf= @identityg;
    case 'threshold',
        threshold = 11;
        params.f     = @(x) (x>threshold);
        params.wordf = @(x) (x>threshold);
        params.df    = @(x) (x>threshold);
        params.worddf= @(x) (x>threshold);    
    case 'tanh',
        params.f = @(x) (tanh(x));
        params.df = @(z) (1-z.^2);
    case 'sclTanh',
        % see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        params.f = @(x) (1.7159*tanh(2/3 * x));
        params.df = @(z) (1.7159 * 2/3 * (1-(2/3 * z).^2));
%     case 'sclTanhTwist',
%         % see http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
%         params.f = @(x) (1.7159*tanh(2/3 * x) + 0.01*x);
%         params.df = @(z) (1.7159 * 2/3 * (1-(2/3 * z).^2) +0.01);
    case 'sigmoid',
        params.f = @(x) (1./(1 + exp(-x)));
        params.df = @(z) (z .* (1 - z));
    case 'rectMax',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f = @(x) (max(0,x));
        params.df = @(z) (z>0);
    case 'relu',
        % see http://www.cs.toronto.edu/~hinton/absps/reluICML.pdf
        params.f = @(x) (max(0,x));
        params.df = @(z) (z>0);
    case 'rectLog1Exp',
        params.f = @(x) (log(1+exp(x)));
        params.df = @(z) (1./(1 + exp(-z)));
    case 'softmax',
        params.f = @(x) (softmax(x));
        %params.df = @(z) (1./(1 + exp(-z)));
    otherwise
        error('Define an activation function!')
end