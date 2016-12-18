function removeLayer(obj)
% Removes a layer from the network.
% 
% George Papamakarios, Feb 2015

% if a loss function is set, removing layers is not allowed
assert(isempty(obj.loss_fun_hl), 'A loss function has been set. Removing layers not allowed.');

% handle net with no layers
assert(obj.num_layers > 0, 'There is no layer to remove.');

% number of parameters of the last layer
num_params_to_rem = obj.num_units(end) * (obj.num_units(end-1) + 1);

% remove the last layer
obj.num_layers  = obj.num_layers - 1;
obj.num_units   = obj.num_units (1:end-1);
obj.num_outputs = obj.num_units (end);
obj.num_params  = obj.num_params - num_params_to_rem;
obj.weights     = obj.weights   (1:end-1);
obj.biases      = obj.biases    (1:end-1);
obj.params      = obj.params    (1:end-num_params_to_rem);
obj.fixed       = obj.fixed     (1:end-num_params_to_rem);
obj.layers      = obj.layers    (1:end-1);

% propagation now has to be done again
obj.done_forwProp = false;
obj.done_backProp = false;
