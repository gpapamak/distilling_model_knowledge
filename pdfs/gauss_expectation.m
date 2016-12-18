function [y] = gauss_expectation(f, m, s2)
% [y] = gauss_expectation(f, m, s2)
% Calculates the expected value E(f) of a function f:R->R under a gaussian
% distribution with mean m and variance s2. m and s2 can be arrays of equal
% size, in which case the function returns E(f) for each (m,s2) element.
%
% George Papamakarios, Jan 2015

assert(isa(f, 'function_handle'), 'Function must be given as a function handle.');
assert(isequal(numel(m), numel(s2)), 'Sizes don''t match.');

integrand = @(x) f(x) * exp(log_gauss_pdf(x, m(:)', s2(:)'));
y = integral(integrand, -inf, inf, 'ArrayValued', true);
