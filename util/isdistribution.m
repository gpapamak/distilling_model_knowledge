function [flag] = isdistribution(p)
% [flag] = isdistribution(p)
% Returns true if p is a valid probability distribution, false otherwise.
%
% George Papamakarios, Jan 2015

tol = 1.0e-12;

p = p(:);

flag = true;
flag = flag && all(p >= 0);
flag = flag && abs(sum(p) - 1) < tol;
