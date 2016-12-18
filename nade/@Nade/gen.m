function [y, x, dLdx] = gen(obj, N)
% Generates samples from nade. Note that the samples are exact. If
% requested, also outputs the derivatives of the log probability at the
% locations of the samples.
% INPUT
%     N     number of samples to generate
% OUTPUT
%     y     conditional probabilities the samples were drawn from
%     x     columns are samples
%     dLdx  columns are derivatives of log p(x)
%
% NOTE: originally I wanted x to be the first argument and y to be the
% second. However, it turns out that y can be more useful as samples, even
% if they are not strictly samples. So I return y first, because that's the
% one I actually use more often.
% 
% George Papamakarios, Jun 2015

y = obj.precision(zeros(obj.num_inputs, N, obj.arraytype));
x = obj.precision(zeros(obj.num_inputs, N, obj.arraytype));
h = obj.precision(zeros(obj.num_inputs, obj.num_hidden, N, obj.arraytype));
W = [obj.W' obj.precision(zeros(obj.num_hidden, 1, obj.arraytype))];

% -- sample

a =  obj.c' * obj.precision(ones(1, N, obj.arraytype));

for i = 1:obj.num_inputs
    
    hi = obj.hidden_f(a);
    y(i,:) = obj.sigm(obj.U(i,:) * hi + obj.b(i));
    x(i,:) = obj.precision(rand(1, N, obj.arraytype) < y(i,:));
    a = a + W(:,i) * x(i,:);
    h(i,:,:) = hi;
end

% -- compute derivatives

if nargout > 2
    
    % derivatives wrt b
    dLdb = repmat(permute(x - y, [1 3 2]), 1, obj.num_hidden, 1);

    % derivatives wrt a
    dLda = dLdb .* repmat(obj.U, 1, 1, N) .* h .* (1 - h);
    dLda = cumsum(dLda(end:-1:2, :, :), 1);

    % derivatives wrt x
    dLdx = sum(dLda(end:-1:1, :, :) .* repmat(obj.W, 1, 1, N), 2);
    dLdx = [permute(dLdx, [1 3 2]); obj.precision(zeros(1, N, obj.arraytype))] + log(y ./ (1-y));
    dLdx = dLdx(obj.rev_order, :);
    
end

y = y(obj.rev_order, :);
x = x(obj.rev_order, :);
