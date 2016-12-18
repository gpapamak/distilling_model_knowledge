function [m, S, Z] = adf_logreg(x, y, tau, method)
% [m, S, Z] = adf_logreg(x, y, tau, method)
% Assumed density filtering for logistic regression.
% INPUT
%     x         DxN matrix with input points as columns
%     y         vector of N labels {-1,+1}
%     tau       variance of the spherical gaussian prior on the weights
%     method    how to approximate integrals (optional)
% OUTPUT
%     m         posterior mean
%     S         posterior covariance matrix
%     Z         normalizing constant
% 
% George Papamakarios, Jan 2015

if nargin < 4
    method = 'mackay';
end

% prepare data
[D, N] = size(x);
y = y(:)';
assert(length(y) == N, 'Sizes don''t match.');
assert(all(y == 1 | y == -1), 'Labels must be either +1 or -1.');
x = x .* (ones(D, 1) * y);

% initialize mean and covariance
m = zeros(D, 1);
S = tau * eye(D, D);

switch method
    case 'monte_carlo'
        [m, S, Z] = adf_logreg_monte_carlo(x, m, S);
    case 'mackay'
        [m, S, Z] = adf_logreg_mackay(x, m, S);
    otherwise
        error('Unknown method.');
end


function [m, S, Z] = adf_logreg_mackay(x, m, S)
% Assumed density filtering for logistic regression using MacKay's 
% approximation to the sigmoid-gaussian integral.

N = size(x, 2);

% initialize normalizing constant
Z = 1;

for n = randperm(N)
           
    % update normalizing constant
    denom2 = 1 + (pi/8) * (x(:,n)' * S * x(:,n));
    Zn = sigm((x(:,n)' * m) / sqrt(denom2));
    Z = Z * Zn;
    
    % alpha and beta parameters
    alpha = (1 - Zn) / sqrt(denom2);
    beta = alpha * (alpha + (pi/8) * (x(:,n)' * m)) / denom2;
    
    % update mean
    m = m + alpha * (S * x(:,n));
    
    % update covariance
    Sx = S * x(:,n);
    S = S - beta * (Sx * Sx');
    S = (S + S') / 2;
    
end


function [m, S, Z] = adf_logreg_monte_carlo(x, m, S)
% Assumed density filtering for logistic regression using monte carlo for
% computing the expectations.

[D, N] = size(x);
nsamples = 1000;

% initialize normalizing constant
Z = 1;

for n = randperm(N)
        
    % draw gaussian samples
    w = gauss_sample(m, S, nsamples);
    
    % calc sigmoid
    this_sigm = sigm(x(:,n)' * w);
    
    % update normalizing constant
    Zn = mean(this_sigm);
    Z = Z * Zn;
    
    % update mean
    m = mean(w .* (ones(D, 1) * this_sigm), 2);
    m = m / Zn;
    
    % update covariance
    w = w - m * ones(1, nsamples);
    S = (w .* (ones(D, 1) * this_sigm)) * w';
    S = S / (nsamples * Zn);
    S = (S + S') / 2;
    
end
