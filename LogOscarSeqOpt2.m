
function [coefs, df, neg_log_likelihood, OrderMatrix, conv] = OscarSeqOpt(tbound, cvalue, Xmatrix, y, start, OrderMatrix)

p = length(Xmatrix(1,:))/2;

sdresponse = std(y);
y = sqrt(p) * y / sdresponse; 
% The rescaling of the response is for computational purposes only (makes the overall variance of y to be p), the
% coefficients will be rescaled back.

start = sqrt(p) * start / sdresponse;
lowbound = zeros(2 * p, 1);

flagvalue = 0;
itervalue = 0;
SolCoef = start(1:p) - start((p + 1):(2 * p));

% The sequential constrained least squares problem is now solved by adding
% an additional constraint at each step and iterating until convergence.

while (flagvalue == 0) && (itervalue < p^2)
    nconstraints = length(OrderMatrix(:, 1));
    A1 = (1 - cvalue) * ones(nconstraints, p) + cvalue * (p * ones(nconstraints, p) - OrderMatrix);
    Amatrix = [A1 A1];
    Bbound = (sqrt(p) * tbound / sdresponse) * ones(1, length(Amatrix(:, 1)));

    % Define the log-likelihood objective function for logistic regression
    objective = @(u) -sum(y .* log(sigmoid(Xmatrix * u)) + (1 - y) .* log(1 - sigmoid(Xmatrix * u)));

    % Optimization options
    options = optimset('Display', 'off', 'LargeScale', 'off', 'Algorithm', 'sqp');

    % Use fmincon to solve the optimization problem
    [x, fval, exitflag] = fmincon(objective, start, [], [], [], [], lowbound, [], @(u) nonlincon(u, Amatrix, Bbound), options);

    SolCoef1 = round((x(1:p) - x((p + 1):(2 * p))) * 10^7) * 10^(-7);
    [currcoef, neworder] = sort(-abs(SolCoef1));
    sameaslast = [0; (currcoef(2:p) == currcoef(1:(p - 1)))];
    startblocksame = [((sameaslast(2:p) - sameaslast(1:(p - 1))) > 0); 0];
    endblocksame = [((sameaslast(2:p) - sameaslast(1:(p - 1))) < 0); sameaslast(p)];
    nblocksame = sum(startblocksame);
    vi = (1:p)';
    visbs = vi(logical(startblocksame));
    viebs = vi(logical(endblocksame));
    for j = 1:nblocksame
        blockmean = mean(vi(visbs(j):viebs(j)));
        vi(visbs(j):viebs(j)) = blockmean * ones(viebs(j) - visbs(j) + 1, 1);
    end
    [tempinvsort, vind] = sort(neworder);
    a1weights = vi(vind)';
    currcoef = -currcoef;
    OrderMatrix = unique([OrderMatrix; a1weights], 'rows');
    flagvalue = 1;
    test = round(SolCoef * 10^7) * 10^(-7) - round(SolCoef1 * 10^7) * 10^(-7);
    SolCoef = SolCoef1;
    flag2 = sqrt(sumsqr(test));
    if (flag2 > 10^(-7))
        flagvalue = 0;
    end
    itervalue = itervalue + 1;
    start = x;
end

coefs = sdresponse * SolCoef / sqrt(p);
df = length(unique(currcoef(currcoef > 0)));
neg_log_likelihood = -fval; % fval is the value of the objective function, which is the negative log-likelihood
conv = ((itervalue < p^2) && (exitflag > 0));

end

% Define the sigmoid function
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end

% Define the nonlinear constraints function
function [c, ceq] = nonlincon(u, Amatrix, Bbound)
    c = Amatrix * u - Bbound'; % Inequality constraints
    ceq = []; % No equality constraints
end
