function [x, y] = gen(obj, N)
% Generates a new data batch of size N.
%    x   input locations
%    y   P(y=1|x)
%
% George Papamakarios, Feb 2015

assert(isint(N) && N > 0, 'Batch size must be a positive integer.');

w = slice_sample(@obj.w_log_post, obj.w_state, N, 'thin', obj.mcmc_thin, 'width', obj.mcmc_slice_width);
x = gauss_sample(obj.x_mean, obj.x_cov, N);

if obj.gen_derivs
    [y1, dy1] = obj.kernel(sum(w .* x, 1));
    y2 = w .* (ones(obj.dim, 1) * dy1);
    y = [y1; y2];
    y = permute(y, [1 3 2]);
else
    y = obj.kernel(sum(w .* x, 1));
end

obj.w_state = w(:,end);
