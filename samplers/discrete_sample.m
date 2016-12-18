function [x] = discrete_sample(p, nsamples, labels)
% [x] = discrete_sample(p, nsamples, labels)
% Samples from a discrete distribution.
% INPUTS
%       p           discrete distribution of length N
%       nsamples    number of samples (optional, defaults to 1)
%       labels      input domain (optional, defaults to {1,2,...,N})
% OUTPUTS
%       x           vector of samples
%
% George Papamakarios, Jan 2015

% prepare distribution
assert(isdistribution(p), 'Probabilities must be non-negative and sum to one.');
p = p(:);
N = length(p);

% prepare labels
if nargin < 3
    labels = 1:N;
end
labels = labels(:);
assert(N == length(labels), 'Sizes don''t match.');

if nargin < 2
    nsamples = 1;
end

% cumulative distribution
c = cumsum(p) * ones(1, nsamples);

% get the samples
r = ones(N, 1) * rand(1, nsamples);
idx = sum(r > c, 1) + 1;
x = labels(idx);
