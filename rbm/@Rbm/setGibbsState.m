function setGibbsState(obj, x)
% Sets the current state of the gibbs chain(s).
% INPUT
%     x   vector(s) to set the state of the chain(s) to. number of columns
%         determine number of states
% 
% George Papamakarios, Jun 2015

assert(ismatrix(x), 'State must be given as a matrix.');
assert(size(x, 1) == obj.num_inputs, sprintf('State must be a vector of length %d.', obj.num_inputs));

if obj.gpu
    obj.gibbs_state = gpuArray(x);
else
    obj.gibbs_state = gather(x);
end

obj.num_chains = size(x, 2);
