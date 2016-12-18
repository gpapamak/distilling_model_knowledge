function [y] = log_mixture_pdf(x, logp, a, w)
% [y] = log_mixture_pdf(x, logp, a, w)
% Log pdf of a mixture of M components. Accepts a set of N input locations 
% and a set of K parameters, and it returns the log pdf evaluated at all 
% KxN combinations.
% INPUT
%       x      DxN matrix, each column is an input location
%       logp   function handle, returns log pdf of a mixture component
%       a      MxK matrix, each column is a list of mixture coefficients
%       w      []xMxK array, each slice is a matrix of parameters for each 
%              of the M mixture components
% OUTPUT
%       y      KxN matrix of log pdf values
%
% George Papamakarios, Jan 2015

N = size(x, 2);
[M, K] = size(a);
assert(size(w, 2) == M && size(w, 3) == K, 'Sizes don''t match.');

y = zeros(K, N);
for k = 1:K
    e = logp(x, w(:,:,k)) + log(a(:,k)) * ones(1, N);
    y(k,:) = logsumexp(e);
end
