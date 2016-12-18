function setParamsFromVec(obj, params, preserve_fixed)
% Set all the parameters of the network (weights and biases) from those
% given as a long vector.
% 
% George Papamakarios, Feb 2015

% if indicated so, don't change the fixed parameters
if preserve_fixed
    params(obj.fixed) = obj.params(obj.fixed);
end

% set weights and biases
i = 1;
for l = 1:obj.num_layers
    
    j = i + numel(obj.weights{l});
    obj.weights{l}(:) = params(i:j-1);
    i = j;
    
    j = i + numel(obj.biases{l});
    obj.biases{l}(:) = params(i:j-1);
    i = j;
end

% set parameter vector
obj.params = params;
