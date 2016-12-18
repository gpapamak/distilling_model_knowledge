function setLogZ(obj, logZ)
% Sets an estimate of the logZ, where Z is the partition function. It is
% then used to normalize the loglikelihood.
% INPUT
%     logZ   an estimate of the log partition function
% 
% George Papamakarios, Jun 2015

assert(isscalar(logZ) && isreal(logZ), 'LogZ must be a real scalar number.');

if obj.gpu
    obj.logZ = gpuArray(logZ);
else
    obj.logZ = gather(logZ);
end
