classdef Rbm < handle
% Implements the binary Restricted Boltzmann Machine.
%
% George Papamakarios, Jun 2015
    
    properties (SetAccess = private, GetAccess = public)
        
        num_inputs = 0
        num_hidden = 0
        num_params = 0
        
        W = []
        a = []
        b = []
        
        logZ = 0
        
        gibbs_state = []
        num_chains = 0
        
        gpu = false
        arraytype = 'double'
        
    end
    
    methods (Access = private, Static)
        
        % logistic sigmoid function, acting elementwise
        function [y] = sigm(x)
            
            % limit the input to avoid underflow
            thres = 36;
            x(x >  thres) =  thres;
            x(x < -thres) = -thres;

            y = 1 ./ (1 + exp(-x));
        end
        
    end
    
    methods (Access = public)
        
        % constructs an rbm with a given number of input and hidden units
        function [obj] = Rbm(num_inputs, num_hidden, platform)
            
            if nargin < 3
                platform = 'cpu';
            end
            
            % check input
            check = @(t) isscalar(t) && isint(t) && t > 0;
            assert(check(num_inputs), 'Number of inputs must be a positive integer.');
            assert(check(num_hidden), 'Number of hidden units must be a positive integer.');
            assert(ismember(platform, {'cpu', 'gpu'}), 'Unknown platform.');
            
            % set the platform
            obj.gpu = strcmp(platform, 'gpu');
            if obj.gpu
                assert(gpuDeviceCount() > 0, 'No gpu found.');
                obj.arraytype = 'gpuArray';
            else
                obj.arraytype = 'double';
            end
            
            % set number of parameters
            obj.num_inputs = num_inputs;
            obj.num_hidden = num_hidden;
            obj.num_params = num_inputs * num_hidden + num_inputs + num_hidden;
            
            % initialize parameters
            obj.W = randn(num_inputs, num_hidden, obj.arraytype) / sqrt((num_inputs + num_hidden) / 2 + 1);
            obj.a = randn(num_inputs, 1, obj.arraytype) / sqrt(num_hidden + 1);
            obj.b = randn(num_hidden, 1, obj.arraytype) / sqrt(num_inputs + 1);
            
            % initialize gibbs state
            obj.gibbs_state = double(rand(obj.num_inputs, 1, obj.arraytype) < 0.5);
            obj.num_chains = 1;
            
        end
        
        % changes the platform to cpu or gpu
        changePlatform(obj, platform)
        
        % set the gibbs state
        setGibbsState(obj, x)
        
        % sets weights and biases
        setParams(obj, W, a, b)
        
        % sets an estimate of logZ
        setLogZ(obj, logZ)
        
        % evaluates unnormalized log likelihood for a given input
        [L, dLdx] = eval(obj, x)
        
        % generates samples
        [x, L, dLdx] = gen(obj, N, thin)
        
        % finds a high probability sample
        [x, L] = max(obj)
        
        % difference from a nade
        [diff, stdev] = difference(obj, nade, type)
        
        % estimate logZ
        [logZ, conf] = estimate_logZ(obj, nade, method, N)
        
        % visualizes the weights as images
        visualize_weights(obj, imsize, varargin)
        
        % visualizes activations of hidden units
        visualize_activations(obj, x)
        
        % creates histograms of parameters
        param_hist(obj);
        
        % checks derivatives of log likelihood wrt input using finite differences
        [err] = checkgrad(obj, Np, Nx, eps)
        
    end
end
