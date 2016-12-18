function clearLossFunction(obj)
% Clears the network's loss function. This allows the addition/removal of
% layers.
% 
% George Papamakarios, May 2015

obj.loss_fun_hl = [];
obj.loss_fun_id = '';
obj.loss_fun_needs_derivs = false;
