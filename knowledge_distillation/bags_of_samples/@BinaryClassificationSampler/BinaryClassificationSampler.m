classdef BinaryClassificationSampler < DataStream
% Full sampler for bayesian binary classification. Each sample consists of 
% an input location x and a parameter vector w. 
%     x is drawn from p(x) = N(x|mx,Sx) using exact sampling
%     y is drawn from p(w|data) using slice sampling
% The probability P(y=1|x,w) is also returned.
%
% George Papamakarios, Feb 2015

    properties (SetAccess = private, GetAccess = public)
        
        data   = []
        dim    = 0
        kernel = @sigm
        
        w_var   = 100
        w_state = []
        
        x_mean = []
        x_cov  = []
        
        mcmc_thin = 1
        mcmc_slice_width = 1
        
    end
    
    methods (Access = private)
        
        % unnormalized log posterior over parameters
        [p] = w_log_post(obj, w)
        
    end
    
    methods (Access = public)
        
        % constructor
        function [obj] = BinaryClassificationSampler(x, y, kernel, w_var, x_mean, x_cov, mcmc_burnin, mcmc_thin, gen_derivs)
        
            % defaults
            if nargin < 9
                gen_derivs = false; % don't generate derivatives
            end
            if nargin < 8
                mcmc_thin = 1; % no thinning
            end
            if nargin < 7
                mcmc_burnin = 0; % no burnin
            end
            
            % check input
            [D, N] = size(x);
            assert(isequal(size(y), [1, N]), 'Sizes don''t match.');
            assert(all(y == 1 | y == -1), 'Labels must be either +1 or -1.');
            assert(isequal(size(x_mean), [D, 1]), 'Sizes don''t match.');
            assert(isequal(size(x_cov), [D, D]), 'Sizes don''t match.');

            % set class properties
            obj.data       = x .* (ones(D, 1) * y);
            obj.dim        = D;
            obj.kernel     = kernel;
            obj.w_var      = w_var;
            obj.x_mean     = x_mean;
            obj.x_cov      = x_cov;
            obj.mcmc_thin  = mcmc_thin;
            obj.gen_derivs = gen_derivs;
            
            % burn in markov chain
            [obj.w_state, obj.mcmc_slice_width] = slice_sample(@obj.w_log_post, zeros(D,1), 'burnin', mcmc_burnin);
        end
        
        % generates a new data batch
        [x, y] = gen(obj, N)
        
    end
end
