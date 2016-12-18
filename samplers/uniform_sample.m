function [x] = uniform_sample(a, nsamples)
% [x] = uniform_sample(a, nsamples)
% Samples from a multivariate rectangular uniform distribution.
% INPUT
%       a          a(:,1) are the lower limits, a(:,2) are the upper limits
%       nsamples   number of samples (optional, defaults to 1)
% OUTPUT
%       x          Dxnsamples matrix where columns are samples
%
% George Papamakarios, Jan 2015

if nargin < 2
    nsamples = 1;
end

D = size(a, 1);
assert(size(a, 2) == 2, 'Limits must have exactly 2 columns.');
assert(all(a(:,2) >= a(:,1)), 'Upper limits must be no less than lower limits.');

x = ((a(:,2) - a(:,1)) * ones(1, nsamples)) .* rand(D, nsamples) + a(:,1) * ones(1, nsamples);
