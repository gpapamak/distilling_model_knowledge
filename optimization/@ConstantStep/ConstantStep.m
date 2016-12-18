classdef ConstantStep < StepStrategy
% Step size strategy where the learning rate is held constant.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        step = []
    end

    methods (Access = public)
        
        % constructor
        function [obj] = ConstantStep(step)
            
            p = inputParser;
            p.addRequired('step', @(t) isscalar(t) && isreal(t) && t > 0);
            p.parse(step);
            
            obj.step = step;
        end
        
        % given current gradient, propose next step
        function [dx] = next(obj, grad)
            
            dx = -obj.step * grad;
        end
        
    end
end
