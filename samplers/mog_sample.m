function [x, z] = mog_sample(a, m, S, nsamples)
% [x] = mog_sample(a, m, S, nsamples)
% Samples from a mixture of K multivariate gaussians.
% INPUT
%       a         vector of K mixing proportions
%       m         DxK matrix, columns are component means
%       S         DxDxK array, slices are component covariances
%       nsamples  number of samples (optional, defaults to 1)
% OUTPUT
%       x         Dxnsamples matrix where columns are samples
%
% George Papamakarios, Jan 2015

if nargin < 4
    nsamples = 1;
end

% check input
K = numel(a);
D = size(m, 1);
if D == 1
    S = permute(S(:), [3 2 1]);
end
assert(size(m, 2) == K && size(S, 3) == K, 'Sizes don''t match.');
assert(size(S, 1) == D && size(S, 2) == D, 'Sizes don''t match.');

% sample which components to use
z = discrete_sample(a, nsamples);

% sample gaussians
x = zeros(D, nsamples);
for k = 1:K
    idx = z == k;
    x(:, idx) = gauss_sample(m(:,k), S(:,:,k), sum(idx));
end
