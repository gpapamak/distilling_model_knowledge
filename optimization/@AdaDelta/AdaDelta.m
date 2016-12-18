classdef AdaDelta < StepStrategy
% ADADELTA step size strategy. For details, see:
% M. D. Zeiler, "ADADELTA: An adaptive learning rate method", arXiv, 2012.
%
% George Papamakarios, Mar 2015

    properties (SetAccess = private, GetAccess = public)
                
        rho = 0.95
        epsilon = 1.0e-6
        
        acc_grad = 0
        acc_dx   = 0
    end

    methods (Access = public)
        
        % constructor
        function [obj] = AdaDelta(rho, epsilon)
            
            if nargin > 1
                assert(epsilon > 0, 'Epsilon must be positive,');
                obj.epsilon = epsilon;
            end    
            if nargin > 0
                assert(0 < rho && rho < 1, 'Rho must be strictly between 0 and 1.');
                obj.rho = rho;
            end            
        end
        
        % given current gradient, propose next step
        function [dx] = next(obj, grad)
            
            obj.acc_grad = obj.rho * obj.acc_grad + (1-obj.rho) * (grad.^2);
            lr = sqrt(obj.acc_dx + obj.epsilon) ./ sqrt(obj.acc_grad + obj.epsilon);
            dx = -lr .* grad;
            obj.acc_dx = obj.rho * obj.acc_dx + (1-obj.rho) * (dx.^2);
        end
        
    end
end
