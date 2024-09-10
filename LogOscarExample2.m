

% 
% % Choose grid of parameter values to use
% % cvalues = [0; .01; .05; .1;.25;.5;.75;.9;1];
% % propvalues = [.0001; .001; .002; .00225; .0025; .00275; .0028; .0029; .003; .004; .005; .0075; .01;.025; .05; .1; .15;.2;.3;.4;.5;.6];
%  cvalues = [0; .01; .05; .1;.25;.5;.75;.9;1];
%  propvalues = [.0001; .001; .002; .00225; .0025; .00275; .0028; .0029; .003; .004; .005;.0075; .01;.025; .05; .1; .15;];
% %%%% method = 2, chooses the sequential algorithm.
% 
% method = 2;     
% initcoef = [];
% [CoefMatrix dfMatrix SSMatrix] = OscarSelect(X, y, cvalues, propvalues, initcoef, method); % Calls function to perform optimization on the grid.
% 



rng(123); % Set seed for reproducibility

n = 200; % Number of observations
p = 10;  % Number of predictors

% Generate 5 highly correlated predictors
rho = 0.9; % Correlation coefficient
Sigma = rho*ones(5) + (1-rho)*eye(5); % Covariance matrix
X1 = mvnrnd(zeros(n,5), Sigma); % Simulate 5 correlated predictors

% Generate 5 uncorrelated predictors
X2 = randn(n, 5);

% Combine all predictors
X = [X1 X2];

% True coefficients (only the first 5 predictors are significant)
beta_true = [2; -2; 1.5; -1.5; 1; zeros(p-5, 1)];

% Generate binary response variable
logit = X * beta_true;
prob = exp(logit) ./ (1 + exp(logit)); % Logistic transformation to get probabilities
y = binornd(1, prob); % Generate binary outcomes based on probabilities

% Convert to table for potential use in the rest of the code
X_table = array2table(X);
y_table = array2table(y);

% Choose grid of parameter values to use
cvalues = [0; .01; .05; .1; .25; .5; .75; .9];
propvalues = [.0001; .001; .002; .00225; .0025; .00275; .0028; .0029; .003; .004; .005; .0075; .01; .025; .05];
method = 2;     
initcoef = [];

% 5-fold cross-validation
K = 5;
indices = crossvalind('Kfold', n, K);
cv_results = zeros(length(cvalues), length(propvalues));

for i = 1:length(cvalues)
    for j = 1:length(propvalues)
        neg_log_likelihoods = zeros(K, 1);
        for k = 1:K
            test_idx = (indices == k);
            train_idx = ~test_idx;
            
            X_train = X(train_idx, :);
            y_train = y(train_idx);
            X_test = X(test_idx, :);
            y_test = y(test_idx);
            
            % Standardize within each fold
            for col = 1:size(X_train, 2)
                X_train(:, col) = (X_train(:, col) - mean(X_train(:, col))) / std(X_train(:, col));
                X_test(:, col) = (X_test(:, col) - mean(X_train(:, col))) / std(X_train(:, col));
            end
            y_train = (y_train - mean(y_train)) / std(y_train);
            y_test = (y_test - mean(y_train)) / std(y_train);
            
            % Ensure no zero variance predictors
            if any(std(X_train) == 0)
                warning('Zero variance predictor found in training set.');
                continue;
            end
            
            [CoefMatrix, ~, NegLogLikelihoodMatrix] = OscarSelect(X_train, y_train, cvalues(i), propvalues(j), initcoef, method);
            
            % Evaluate on test set
            test_coefs = CoefMatrix(:, 1, 1); % Assuming the same indexing structure as before
            pred = X_test * test_coefs;
            
            % Add small regularization term to avoid log(0) issues
            epsilon = 1e-10;
            neg_log_likelihoods(k) = -sum(y_test .* log(sigmoid(pred) + epsilon) + (1 - y_test) .* log(1 - sigmoid(pred) + epsilon));
        end
        cv_results(i, j) = mean(neg_log_likelihoods);
    end
end

% Find the best hyperparameters
[min_cv, idx] = min(cv_results(:));
[i, j] = ind2sub(size(cv_results), idx);

best_propvalue = propvalues(j);
best_cvalue = cvalues(i);

% Retrain the model on the entire dataset using the best hyperparameters
[CoefMatrix, dfMatrix, NegLogLikelihoodMatrix] = OscarSelect(X, y, best_cvalue, best_propvalue, initcoef, method);

% Find the model with the lowest negative log-likelihood
[minNegLogLikelihood, idx] = min(NegLogLikelihoodMatrix(:));
[best_i, best_j, best_k] = ind2sub(size(NegLogLikelihoodMatrix), idx);

% Extract corresponding degrees of freedom, propvalues, and cvalues
best_df = dfMatrix(best_i, best_j, best_k);
best_coefs = CoefMatrix(:, best_j, best_k);

% Print the results
fprintf('Best Model (after cross-validation):\n');
fprintf('Degrees of Freedom: %d\n', best_df);
fprintf('propvalue: %g\n', best_propvalue);
fprintf('cvalue: %g\n', best_cvalue);
fprintf('Negative Log-Likelihood: %g\n', minNegLogLikelihood);
fprintf('Coefficients:\n');
disp(best_coefs);

% Define the sigmoid function
function g = sigmoid(z)
    g = 1 ./ (1 + exp(-z));
end