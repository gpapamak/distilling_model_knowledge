function forwProp(obj, x)
% Forward propagation in the network for specified input.
% INPUT
%     x   columns are input locations
% 
% George Papamakarios, Feb 2015

N = size(x, 2);
obj.layers{1}.x = x;

for l = 1:obj.num_layers
    z = obj.weights{l} * obj.layers{l}.x + obj.biases{l} * ones(1, N, obj.arraytype);
    obj.layers{l+1}.forwProp(z);
end

obj.done_forwProp = true;
obj.done_backProp = false;
