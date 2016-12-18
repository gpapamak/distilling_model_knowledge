classdef NadeEnsemble < handle
% An ensemble of nades.
%
% George Papamakarios, Jun 2015
    
    properties (SetAccess = private, GetAccess = public)
        
        nades = {}
        arraytype = 'double'
        platform = 'cpu'
        
        num_nades  = 0
        num_inputs = 0
        num_params = 0

    end
    
    methods (Access = public)
        
        % constructs an ensemble given a set of nades
        function [obj] = NadeEnsemble(nades, platform)
            
            if nargin < 2
                platform = 'cpu';
            end
            switch platform
                case 'cpu'
                    obj.arraytype = 'double';
                case 'gpu'
                    obj.arraytype = 'gpuArray';
                otherwise
                    error('Unknown platform.');
            end
            obj.platform = platform;
            
            assert(iscell(nades), 'Input must be a cell array of nades.');
            assert(~isempty(nades), 'Input can''t be empty.');
            
            obj.num_nades = numel(nades);
            
            assert(isa(nades{1}, 'Nade'), 'Input must be a cell array of nades.');
            obj.num_inputs = nades{1}.num_inputs;
            obj.num_params = nades{1}.num_params;
            nades{1}.changePlatform(platform);
            
            for i = 2:obj.num_nades
                assert(isa(nades{i}, 'Nade'), 'Input must be a cell array of nades.');
                assert(nades{i}.num_inputs == obj.num_inputs,  'All nades must have the same number of inputs.');
                obj.num_params = obj.num_params + nades{i}.num_params;
                nades{i}.changePlatform(platform);
            end
            
            obj.nades = nades(:);
        end
        
        % evaluates ensemble for a given input
        function [L, dLdx] = eval(obj, x)
            
            L = 0;

            if nargout > 1

                dLdx = 0;

                for i = 1:obj.num_nades
                    [Li, ~, dLidx] = obj.nades{i}.eval(x);
                    Li = exp(Li);
                    L = L + Li;
                    dLdx = dLdx + (ones(obj.num_inputs, 1, obj.arraytype) * Li) .* dLidx;
                end
                dLdx = dLdx ./ (ones(obj.num_inputs, 1, obj.arraytype) * L);

            else
                for i = 1:obj.num_nades
                    L = L + exp(obj.nades{i}.eval(x));
                end
            end

            L = log(L) - log(obj.num_nades); 
        end
        
        % generates samples from ensemble
        function [x, L, dLdx] = gen(obj, N)
           
            x = zeros(obj.num_inputs, N, obj.arraytype);
            nades_idx = randi(obj.num_nades, [1 N], obj.arraytype);
            
            for i = 1:obj.num_nades
                nn = nades_idx == i;
                [~, x(:,nn)] = obj.nades{i}.gen(sum(nn));
            end
            
            if nargout == 3
                [L, dLdx] = obj.eval(x);
            elseif nargout == 2
                L = obj.eval(x);
            end
        end
        
        % difference between this and another nade
        function [diff, stdev] = difference(obj, nade, type)

            assert(isa(nade, 'Nade') || isa(nade, 'NadeEnsemble'), 'Other model must be a nade or a nade ensemble.');

            N = 500;
            [x, thisL] = obj.gen(N);
            othrL = nade.eval(x);

            switch type

                case 'kl'

                    err = thisL - othrL;
                    diff = mean(err);
                    stdev = std(err) / sqrt(N);

                case 'square_error'

                    err = (thisL - othrL) .^ 2;
                    diff = mean(err) / 2;
                    stdev = std(err) / (2 * sqrt(N));

                otherwise
                    error('Unknown difference measure.');
            end
        end

        % changes the platform to cpu or gpu
        function changePlatform(obj, platform)
            
            switch platform
                case 'cpu'
                    obj.arraytype = 'double';
                case 'gpu'
                    obj.arraytype = 'gpuArray';
                otherwise
                    error('Unknown platform.');
            end
            obj.platform = platform;
            
            for i = 1:obj.num_nades
                obj.nades{i}.changePlatform(platform);
            end
                       
        end
        
    end
end
