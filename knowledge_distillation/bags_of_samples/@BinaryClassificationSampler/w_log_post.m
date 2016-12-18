function [p] = w_log_post(obj, w)
% Unnormalized log posterior over parameters:
%    logP(w,data) = logP(w) + sumlogP(y|x,w)
%
% George Papamakarios, Feb 2015

p = obj.kernel(w' * obj.data);
p = sum(log(p), 2);
p = p - sum(w' .^ 2, 2) / (2 * obj.w_var);
