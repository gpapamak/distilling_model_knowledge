function [a, m, S, z, loglik, iter] = em_mog(x, K, varargin)
% [a, m, S, z, loglik, iter] = em_mog(x, K, varargin)
% Maximum likelihood learning of a mixture of multivariate gaussians using
% expectation-maximization.
% INPUT
%    x          DxN matrix, each column is an observed variable
%    K          number of gaussian components
%    -- optional name-value pairs --
%    verbose    if true, print out information during execution
%    tol        if diff in log-likelihood less than tol, terminate
%    maxiter    maximum number of iterations
%    init_a     initial value for mixing coefficients
%    init_m     initial value for means
%    init_S     initial value for covariances
% OUTPUT
%    a          learnt mixing coefficients
%    m          DxK array, learnt means
%    S          DxDxK array, learnt convariances
%    z          KxN array, z(n,k) = prob(mixture k | variable n)
%    loglik     final value of log likelihood = prob(x | a, m, S)
%    iter       number of iterations till convergence
%
% George Papamakarios, Jan 2015

[D, N] = size(x);

p = inputParser;
p.addRequired('x', @(t) ismatrix(t) && isreal(t));
p.addRequired('K', @(t) isscalar(t) && isint(t) && t > 0);
p.addParameter('verbose', false, @(t) isscalar(t) && islogical(t));
p.addParameter('tol', 1.0e-7, @(t) isscalar(t) && t >= 0);
p.addParameter('maxiter', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('init_a', ones(K, 1) / K, @(t) isdistribution(t) && numel(t) == K);
p.addParameter('init_m', randn(D, K), @(t) isreal(t) && isequal(size(t), [D, K]));
p.addParameter('init_S', repmat(eye(D,D), [1 1 K]), @(t) isreal(t) && isequal(size(t), [D, D, K]));
p.parse(x, K, varargin{:});

% initialize
a = p.Results.init_a(:);
m = p.Results.init_m;
S = p.Results.init_S;
loglik = -inf;
diff = inf;
iter = 0;

while diff > p.Results.tol && iter < p.Results.maxiter

    % E-step
    z = log_gauss_pdf(x, m, S);
    z = z + log(a) * ones(1, N);
    z = z - ones(K, 1) * logsumexp(z);
    z = exp(z);
    
    % M-step
    Nk = sum(z, 2);
    a = Nk / N;
    m = (x * z') / diag(Nk);
    for k = 1:K
        xm = x - m(:,k) * ones(1, N);
        S(:,:,k) = (xm * diag(z(k,:)) * xm') / Nk(k);
    end
    
    % check progress
    iter = iter + 1;
    loglik_new = sum(log_mog_pdf(x, a, m, S));
    diff = loglik_new - loglik;
    assert(diff >= 0, 'Log-likelihood decreased! There is a bug somewhere!');
    loglik = loglik_new;
    
    % show info
    if p.Results.verbose
        fprintf('Iteration %d, loglik = %g, diff = %g \n', iter, loglik, diff);
    end

end
