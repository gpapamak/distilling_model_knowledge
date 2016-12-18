function [a, m, S, z] = em_mog_stepwise(x, a, m, S, step)
% [a, m, S, z] = em_mog_stepwise(x, a, m, S, step)
% Maximum likelihood learning of a mixture of multivariate gaussians using
% online expectation-maximization, stepwise variant. Given a minibatch x,
% it performs a single update on the parameters.
% INPUT
%    x        DxN matrix, each column is an observed variable
%    a        K previous mixing coefficients
%    m        DxK array, previous means
%    S        DxDxK array, previous convariances
%    step     update stepsize (0 <= step <= 1)
% OUTPUT
%    a, m, S  updated parameters
%    z        KxN array, z(n,k) = prob(mixture k | variable n)
%
% George Papamakarios, Jan 2015

a = a(:);
K = length(a);
[D, N] = size(x);
assert(isequal(size(m), [D, K]), 'Sizes don''t match.');
assert(isequal(size(S), [D, D, K]), 'Sizes don''t match.');
assert(0 <= step && step <= 1, 'Stepsize must be in [0,1].');

% E-step
z = log_gauss_pdf(x, m, S);
z = z + log(a) * ones(1, N);
z = z - ones(K, 1) * logsumexp(z);
z = exp(z);

% M-step (sufficient statistics)
phi_a = mean(z, 2);
phi_m = (x * z') / N;
phi_S = zeros(D, D, K);
for k = 1:K
    phi_S(:,:,k) = (x * diag(z(k,:)) * x') / N;
end

% calculate old sufficient_statistics
phi_a_old = a;
phi_m_old = m * diag(a);
phi_S_old = zeros(D, D, K);
for k = 1:K
    phi_S_old(:,:,k) = a(k) * (S(:,:,k) + m(:,k) * m(:,k)');
end

% update sufficient statistics
phi_a = (1-step) * phi_a_old + step * phi_a;
phi_m = (1-step) * phi_m_old + step * phi_m;
phi_S = (1-step) * phi_S_old + step * phi_S;

% calculate new parameters
a = phi_a;
m = phi_m / diag(a);
for k = 1:K
    S(:,:,k) = phi_S(:,:,k) / a(k) - m(:,k) * m(:,k)';
end
