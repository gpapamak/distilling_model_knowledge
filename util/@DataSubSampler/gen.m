function [varargout] = gen(obj, N)
% Generates a new data batch of size N from the data set.
%
% George Papamakarios, Feb 2015

assert(isint(N) && N > 0, 'Batch size must be a positive integer.');

varargout = cell(obj.num_mats, 1);
idx = cell(obj.num_mats, 1);
for k = 1:obj.num_mats
    sizk = size(obj.x{k});
    sizk(end) = N;
    varargout{k} = zeros(sizk, class(obj.x{k}));
    idx{k} = repmat({':'}, 1, ndims(obj.x{k})-1);
end
a = 1;

j = obj.i + N - 1;
times = floor(j / obj.num_data);
new_i = mod(j, obj.num_data);

for t = 1:times
    
    n = obj.nn(obj.i : end);
    b = a + length(n) - 1;
    
    for k = 1:obj.num_mats
        varargout{k}(idx{k}{:}, a : b) = obj.x{k}(idx{k}{:}, n);
    end
    a = b + 1;
    
    % reshuffle data
    obj.nn = randperm(obj.num_data);
    obj.i = 1;
    
end

n = obj.nn(obj.i : new_i);
for k = 1:obj.num_mats
    varargout{k}(idx{k}{:}, a : N) = obj.x{k}(idx{k}{:}, n);
end

obj.i = new_i + 1;
