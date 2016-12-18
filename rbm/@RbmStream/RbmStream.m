classdef RbmStream < DataStream
% Given an rbm, it generates minibatches by sampling from it. Optionally,
% it also returns the unnormalized log probability and its derivatives wrt
% the inputs at the sample locations.
%
% George Papamakarios, Jun 2015

    properties (SetAccess = private, GetAccess = public)
        
        rbm = []
        thin = 1
        idx = 1
        
    end
    
    methods (Access = public)
        
        % constructor
        function [obj] = RbmStream(rbm, burnin, thin)

            if nargin < 3
                thin = 1;
            end
            if nargin < 2
                burnin = 0;
            end
            
            assert(isa(rbm, 'Rbm'), 'Input has to be an rbm.');
            assert(isscalar(thin) && isint(thin) && thin > 0, 'Thinning amount must be a positive integer.');
            assert(isscalar(burnin) && isint(burnin) && burnin >= 0, 'Burnin must be a non-negative integer.');
            
            obj.rbm = rbm;
            obj.thin = thin;
            
            if burnin > 0
                obj.rbm.gen(obj.rbm.num_chains, burnin);
            end
        end
        
        % generates a new data batch
        function [varargout] = gen(obj, N)
            
            assert(isscalar(N) && isint(N) && N > 0, 'Batch size must be a positive integer.');
            nargoutchk(1, 3);
            
            ii = obj.idx : obj.idx + N - 1;
            ii = mod(ii - 1, obj.rbm.num_chains) + 1;
            obj.idx = mod(ii(end), obj.rbm.num_chains) + 1;
            
            switch nargout
                case 3
                    x = obj.rbm.gen(obj.rbm.num_chains, obj.thin);
                    x = x(:,ii);
                    [L, dLdx] = obj.rbm.eval(x);
                    varargout{1} = x;
                    varargout{2} = L;
                    varargout{3} = dLdx;
                
                case 2
                    x = obj.rbm.gen(obj.rbm.num_chains, obj.thin);
                    x = x(:,ii);
                    L = obj.rbm.eval(x);
                    varargout{1} = x;
                    varargout{2} = L;
                                    
                case 1
                    x = obj.rbm.gen(obj.rbm.num_chains, obj.thin);
                    varargout{1} = x(:,ii);
            end
        end
        
    end
end
