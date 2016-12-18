classdef NeuralNet < handle
% Implements a feedworward neural network.
% Supports various types of layers and loss functions.
%
% George Papamakarios, Feb 2015
    
    properties (SetAccess = private, GetAccess = public)
        
        gpu = false
        arraytype = 'double'
        
        num_layers  = 0
        num_units   = 1
        num_inputs  = 1
        num_outputs = 1
        num_params  = 0
        
        weights = {}
        biases  = {}
        params  = []
        fixed   = logical([])
        layers  = {}
        
        done_forwProp = false
        done_backProp = false
        
        dydp = []
        Hpvx = []
        Hxvx = []
        
        loss_fun_hl = []
        loss_fun_id = ''
        loss_fun_needs_derivs = false
        
    end
    
    methods (Access = private)
       
        % sets network parameters from a long vector
        setParamsFromVec(obj, params, preserve_fixed)
        
        % forward propagation
        forwProp(obj, x)
        
        % backward propagation
        backProp(obj, dydx)
        
        % backward propagation for inputs only
        backProp_inputs(obj, dydx)
        
        % R{backward propagation}
        RbackProp(obj, vx, d2ydx2)
        
        % clears all intermediate results
        clear(obj)
        
        % square error loss
        [L, dLdp] = square_error(obj, x, y)
        
        % cross entropy loss
        [L, dLdp] = cross_entropy(obj, x, y)
        
        % multilabel cross entropy loss
        [L, dLdp] = multi_cross_entropy(obj, x, y)
        
        % dot product loss
        [L, dLdp] = dot_product(obj, x, y)
        
        % average score matching loss
        [L, dLdp] = avg_score_matching(obj, x, y)
        
        % square error of derivatives loss
        [L, dLdp] = deriv_square_error(obj, x, y)
        
        % square error & square error of derivatives
        [L, dLdp] = square_error_and_deriv_square_error(obj, x, y, lambda)
        
        % cross entropy & square error of derivatives
        [L, dLdp] = cross_entropy_and_deriv_square_error(obj, x, y, lambda)
        
    end
    
    methods (Access = public)
        
        % constructs a net with a given number of inputs and no layers
        function [obj] = NeuralNet(num_inputs, platform)
            
            if nargin < 2
                platform = 'cpu';
            end
            
            switch platform
                case 'cpu'
                    obj.gpu = false;
                    obj.arraytype = 'double';
                case 'gpu'
                    assert(gpuDeviceCount() > 0, 'No gpu found.');
                    obj.gpu = true;
                    obj.arraytype = 'gpuArray';
                otherwise
                    error('Unknown platform.');
            end
            
            check = @(t) isscalar(t) && isint(t) && t > 0;
            assert(check(num_inputs), 'Number of inputs must be a positive integer.');
            
            obj.num_units   = num_inputs;
            obj.num_inputs  = num_inputs;
            obj.num_outputs = num_inputs;
            
            obj.layers = {struct()};
        end
        
        % adds a layer to the network
        addLayer(obj, num_units, varargin)
        
        % removes a layer from the network
        removeLayer(obj)
        
        % change the nonlinearity of a layer
        changeLayerType(obj, layer, newtype)
        
        % changes the platform
        changePlatform(obj, platform)
        
        % sets loss function to use
        setLossFunction(obj, loss_fun, lambda)
        
        % clears loss function
        clearLossFunction(obj)
        
        % evaluates network for a given input
        [y, dydx] = eval(obj, x)
        
        % evaluates loss function
        [L, dLdp] = eval_loss(obj, x, y)
        
        % checks the networks's derivatives using finite differences
        [err, err_p, err_x] = checkgrad(obj, Np, Nx, eps)
        
        % checks the networks's hessian using finite differences
        [err, err_p, err_x] = checkhess(obj, Np, Nx, Nv, eps)
        
        % checks the derivatives of the loss function using finite differences
        [err] = checkloss(obj, Np, Nx, Nd, eps)
        
        % trains the network given a set of examples
        [loss, iter, trace] = train(obj, x, y, varargin)
        
        % trains the network given a data stream
        [loss, iter, trace] = train_stream(obj, stream, varargin)
        
        % trains the network given another network
        [loss, iter, trace] = train_net(obj, net, inp, varargin)
        
        % visualizes the weights of a specific layer as images
        visualize_weights(obj, layer, imsize, varargin)
        
        % visualizes activations for specified layers
        visualize_activations(obj, x, layers)
        
        % creates histograms of weights and biases
        param_hist(obj, layers);
        
    end
end
