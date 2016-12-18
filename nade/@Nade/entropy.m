function [H, err] = entropy(obj, ns)
% Estimates the entropy of nade using monte carlo. Note that the entropy is
% measured in nats.
% INPUT
%     ns      number of samples to use for the estimate (optional)
% OUTPUT
%     H       entropy estimate
%     err     estimate of the standard deviation of H
% 
% George Papamakarios, Jun 2015

if nargin < 2
    ns = 1000;
end

% generate samples from nade
y = obj.gen(ns);

% form monte carlo estimate
L = sum(y .* log(y) + (1-y) .* log(1-y), 1);
H = -mean(L);
err = std(L) / sqrt(ns);
