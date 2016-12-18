function clear(obj)
% Clears the network of all intermediate results.
%
% George Papamakarios, Mar 2015

obj.layers{1} = struct();

for l = 1:obj.num_layers
    obj.layers{l+1}.clear();
end

obj.dydp = [];
obj.Hpvx = [];
obj.Hxvx = [];

obj.done_forwProp = false;
obj.done_backProp = false;
