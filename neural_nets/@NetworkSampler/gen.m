function [x, y] = gen(obj, N)
% Generates a new data batch of size N.
%    x   input locations
%    y   network's outputs
%
% George Papamakarios, Feb 2015

assert(isint(N) && N > 0, 'Batch size must be a positive integer.');

x = obj.input_sampler(N);

if obj.gen_derivs
    [y1, y2] = obj.net.eval(x);
    y = [permute(y1, [3 1 2]); y2];
else
    y = obj.net.eval(x);
end
