function setParams(obj, W, a, b)
% Sets the weights and the biases of the RBM.
% INPUT
%     W     weight matrix
%     a     visible biases
%     b     hidden biases
% 
% George Papamakarios, Jun 2015

assert(isequal(size(W), [obj.num_inputs, obj.num_hidden]), sprintf('Weight matrix must be of size %d x %d.', obj.num_inputs, obj.num_hidden));
assert(isequal(size(a), [obj.num_inputs, 1]), sprintf('Visible biases must be a column vector of size %d.', obj.num_inputs));
assert(isequal(size(b), [obj.num_hidden, 1]), sprintf('Hidden biases must be column vector of size %d.', obj.num_hidden));

if obj.gpu
    f = @gpuArray;
else
    f = @gather;
end

obj.W = f(W);
obj.a = f(a);
obj.b = f(b);
