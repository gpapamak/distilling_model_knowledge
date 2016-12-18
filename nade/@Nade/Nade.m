classdef Nade < handle
% Implements binary NADE.
%
% Reference:
% H. Larochelle and I. Murray, "The Neural Autoregressive Distribution
% Estimator", JMLR, 2011. (corrected version)
%
% George Papamakarios, Jun 2015
    
    properties (SetAccess = private, GetAccess = public)
        
        type = []
        
        gpu = false
        arraytype = 'double'
        precision = @double;
        
        num_inputs = 0
        num_hidden = 0
        num_params = 0
        
        fwd_order = []
        rev_order = []
        
        W = []
        c = []
        U = []
        b = []
        
        x = []
        h = []
        y = []
        L = []
        
        dLdW = []
        dLdc = []
        dLdU = []
        dLdb = []
        
        dLdx = []
        dLda = []
        
        RdLdW = []
        RdLdc = []
        RdLdU = []
        RdLdb = []
        
        done_forwProp = false
        done_backProp = false
        
        hidden_f = []
        backProp = []
        backProp_inputs = []
        RbackProp = []
        
    end
    
    methods (Access = private, Static)
        
        % logistic sigmoid function, acting elementwise
        function [y] = sigm(x)
            
            % limit the input to avoid underflow
            if isa(x, 'single')
                thres = 16;
            else
                thres = 36;
            end
            
            x(x >  thres) =  thres;
            x(x < -thres) = -thres;

            y = 1 ./ (1 + exp(-x));
        end
        
    end
    
    methods (Access = private)
       
        % sets nade's parameters from a long vector
        setParamsFromVec(obj, params)
        
        % forward propagation
        forwProp(obj, x)
        
        % forward propagation where the input is generated by nade
        forwProp_gen(obj, N)
        
        % backward propagation (logistic version)
        backProp_logistic(obj)
                
        % backward propagation for inputs only (logistic version)
        backProp_logistic_inputs(obj)
        
        % R{backprop} (logistic version)
        RbackProp_logistic(obj, vx)
        
        % backward propagation (relu version)
        backProp_relu(obj)
                
        % backward propagation for inputs only (relu version)
        backProp_relu_inputs(obj)
        
        % reduced version of backprop for inputs, assuming backprop for
        % parameters has been run already
        backProp_inputs_reduced(obj)
        
        % R{backprop} (relu version)
        RbackProp_relu(obj, vx)
        
        % clears all intermediate results
        clear(obj)
        
    end
    
    methods (Access = public)
        
        % constructs a nade with a given number of input and hidden units
        function [obj] = Nade(num_inputs, num_hidden, type, platform, order, precision)
            
            if nargin < 6
                precision = 'double';
            end
            if nargin < 4
                platform = 'cpu';
            end
            
            % check input
            check = @(t) isscalar(t) && isint(t) && t > 0;
            assert(check(num_inputs), 'Number of inputs must be a positive integer.');
            assert(check(num_hidden), 'Number of hidden units must be a positive integer.');
            assert(ismember(type, {'logistic', 'relu'}), 'Unknown hidden layer type.');
            assert(ismember(platform, {'cpu', 'gpu'}), 'Unknown platform.');
            assert(ismember(precision, {'single', 'double'}), 'Unknown precision.');
            
            % set precision
            switch precision
                case 'single'
                    obj.precision = @single;
                case 'double'
                    obj.precision = @double;
            end
            
            % if nade is on gpu, make the necessary arrangements
            obj.gpu = strcmp(platform, 'gpu');
            if obj.gpu
                assert(gpuDeviceCount() > 0, 'No gpu found.');
                createArray = @gpuArray;
                obj.arraytype = 'gpuArray';
            else
                createArray = @(x) x;
                obj.arraytype = 'double';
            end
            
            % set number of parameters
            obj.num_inputs = num_inputs;
            obj.num_hidden = num_hidden;
            obj.num_params = num_inputs * (2*num_hidden + 1);
            
            % set the ordering
            if nargin < 5
                obj.fwd_order = obj.precision(createArray(randperm(num_inputs)));
            else
                obj.fwd_order = obj.precision(createArray(order));
            end
            [~, obj.rev_order] = sort(obj.fwd_order);
            
            % initialize parameters
            obj.W = obj.precision(randn(num_inputs - 1, num_hidden, obj.arraytype) / sqrt(num_inputs));
            obj.c = obj.precision(randn(1, num_hidden, obj.arraytype) / sqrt(num_inputs));
            obj.U = obj.precision(randn(num_inputs, num_hidden, obj.arraytype) / sqrt(num_hidden + 1));
            obj.b = obj.precision(randn(num_inputs, 1, obj.arraytype) / sqrt(num_hidden + 1));
                        
            % set up hidden layer
            switch type
                case 'logistic'
                    obj.hidden_f = @obj.sigm;
                    obj.backProp = @obj.backProp_logistic;
                    obj.backProp_inputs = @obj.backProp_logistic_inputs;
                    obj.RbackProp = @obj.RbackProp_logistic;
                    
                case 'relu'
                    obj.hidden_f = @(x) max(0, x);
                    obj.backProp = @obj.backProp_relu;
                    obj.backProp_inputs = @obj.backProp_relu_inputs;
                    obj.RbackProp = @obj.RbackProp_relu;
            end
            obj.type = type;
                    
        end
        
        % changes the platform to cpu or gpu
        changePlatform(obj, platform)
        
        % evaluates output for a given input
        [L, y, dLdx] = eval(obj, x)
        
        % generates samples from nade
        [y, x, dLdx] = gen(obj, N)
        
        % estimates the entropy of nade using monte carlo
        [H, err] = entropy(obj, ns)
        
        % finds a high probability sample
        [x, L] = max(obj)
        
        % difference from another model
        [diff, stdev] = difference(obj, model, type)
        
        % checks nade's derivatives using finite differences
        [err, err_p, err_x] = checkgrad(obj, Np, Nx, eps)
        
        % checks nade's hessian using finite differences
        [err] = checkhess(obj, Np, Nx, Nv, eps)
        
        % trains nade given a set of samples
        [progress] = train(obj, x, varargin)
        
        % trains nade given another nade
        [progress] = train_nade(obj, nade, varargin)
        
        % trains nade given a data stream
        [progress] = train_stream(obj, stream, varargin)
        
        % visualizes the weights of a specific layer as images
        visualize_weights(obj, layer, imsize, varargin)
        
        % visualizes activations for specified layers
        visualize_activations(obj, x)
        
        % creates histograms of parameters
        param_hist(obj);
        
    end
end