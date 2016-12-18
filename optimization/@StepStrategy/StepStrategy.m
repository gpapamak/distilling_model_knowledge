classdef StepStrategy < handle
% Abstract class for the step size strategy of stochastic gradient
% training.
%
% George Papamakarios, Mar 2015

    methods (Abstract, Access = public)
        
        % given current gradient, propose next step
        [dx] = next(obj, grad)
        
    end
end
