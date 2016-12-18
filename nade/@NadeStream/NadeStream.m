classdef NadeStream < DataStream
% Given a nade, it generates minibatches by sampling from it. Optionally,
% it also returns the log probability and its derivatives wrt the inputs,
% at the sample locations.
%
% George Papamakarios, Jun 2015

    properties (SetAccess = private, GetAccess = public)
        
        nade = []
        
    end
    
    methods (Access = public)
        
        % constructor
        function [obj] = NadeStream(nade)

            assert(isa(nade, 'Nade'), 'Input has to be a nade.');
            obj.nade = nade;
        end
        
        % generates a new data batch
        function [varargout] = gen(obj, N)
            
            assert(isint(N) && N > 0, 'Batch size must be a positive integer.');
            nargoutchk(1, 3);
            
            switch nargout
                case 3
                    [y, x, dLdx] = obj.nade.gen(N);
                    varargout{1} = x;
                    varargout{2} = sum(x .* log(y) + (1-x) .* log(1-y), 1);
                    varargout{3} = dLdx;
                
                case 2
                    [y, x] = obj.nade.gen(N);
                    varargout{1} = x;
                    varargout{2} = sum(x .* log(y) + (1-x) .* log(1-y), 1);
                                    
                case 1
                    [~, x] = obj.nade.gen(N);
                    varargout{1} = x;
            end
        end
        
    end
end
