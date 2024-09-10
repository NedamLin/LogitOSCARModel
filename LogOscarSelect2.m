

function [CoefMatrix, dfMatrix, NegLogLikelihoodMatrix] = OscarSelect(X, y, cvalues, propvalues, initcoef, method)

p = length(X(1,:));

% Standardize predictors to mean zero, variance 1, and center response to
% mean zero.

for i = 1:p
  X(:,i) = (X(:,i)-mean(X(:,i)))/std(X(:,i));
end
y = y - mean(y);

if nargin < 6
    method = 2;
    if nargin < 5
        initcoef = [];
    end
end

cvalues = sort(cvalues);
propvalues = sort(propvalues);

if isempty(initcoef)
    % negative log-likelihood function
    negLogLikelihood = @(params) -sum(y .* log(sigmoid(X * params)) + (1 - y) .* log(1 - sigmoid(X * params)));
    
    % Initial guess for the parameters
    initialParams = zeros(p, 1);
    
    % Optimization options
    options = optimset('Display', 'off', 'LargeScale', 'off');
    
    % Use fminunc to find the parameters that minimize the negative log-likelihood
    initcoef = fminunc(negLogLikelihood, initialParams, options);
end

if length(initcoef) < p
    error('initial estimate must be of length p');
end
if min(cvalues) < 0
    error('all values of c must be nonnegative');
end
if max(cvalues) > 1
    error('values for c cannot exceed 1');
end
if min(propvalues) <= 0
    error('values for proportion must be greater than 0');
end
if max(propvalues) >= 1
    error('values for proportion must be smaller than 1');
end

if method == 1
    [CoefMatrix, dfMatrix, NegLogLikelihoodMatrix] = OscarReg(X, y, cvalues, propvalues, initcoef);
else
    [CoefMatrix, dfMatrix, NegLogLikelihoodMatrix] = OscarSeqReg(X, y, cvalues, propvalues, initcoef);
end

end

function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end
