function [y] = logdet(x, option)
% [y] = logdet(x)
% Numerically stable log of the determinant of a matrix x. In contrast, 
% matlab's log(det(x)) can easily overflow to inf or underflow to -inf for 
% large x.
% If option = 'pd', matrix x is assumed to be strictly positive definite
% and the fast cholesky decomposition is used. Otherwise, x can be any
% square matrix and the slower eigen decomposition is used.
%
% George Papamakarios, Jan 2015

if nargin < 2 || ~strcmp(option, 'pd')
    
    % any square matrix with non-negative determinant
    L = eig(x);
    neg = L < 0;
    assert(mod(sum(neg), 2) == 0, 'Matrix must have non-negative determinant.');
    L(neg) = abs(L(neg));
    y = sum(log(L));
    
else
    
    % strictly positive definite matrix
    y = 2 * sum(log(diag(chol(x))));
    
end
