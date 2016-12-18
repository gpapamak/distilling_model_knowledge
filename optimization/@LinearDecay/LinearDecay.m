classdef LinearDecay < StepStrategy
% Step size strategy where the learning rate is linearly decreased so as to
% hit zero after a specified number of iterations.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        iter = 0
        init = []
        maxiter = []
    end

    methods (Access = public)
        
        % constructor
        function [obj] = LinearDecay(init, maxiter)
            
            p = inputParser;
            p.addRequired('init', @(t) isscalar(t) && isreal(t) && t > 0);
            p.addRequired('maxiter', @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
            p.parse(init, maxiter);
            
            obj.init = init;
            obj.maxiter = maxiter;
        end
        
        % given current gradient, propose next step
        function [dx] = next(obj, grad)
            
            lr = obj.init * (1 - obj.iter / obj.maxiter);
            dx = -lr * grad;
            obj.iter = obj.iter + 1;
        end
        
    end
end
