classdef DataSubSampler < DataStream
% Given a data set, subsamples mini-batches from it.
%
% George Papamakarios, Feb 2015

    properties (SetAccess = private, GetAccess = public)
        
        num_data = 0
        num_mats = 0
        
        x = {}
        
        nn = []
        i = 0;
        
    end
    
    methods (Access = public)
        
        % constructor
        function [obj] = DataSubSampler(varargin)
        
            % check number of inputs
            narginchk(1, inf);
            obj.num_mats = nargin;
            obj.x = varargin(:);
            
            % check input matrices
            check = @(t) isreal(t) && ~isempty(t);
            assert(check(obj.x{1}), 'Data must be given as real nonempty arrays.');
            N = size(obj.x{1}, ndims(obj.x{1}));
            for k = 2:obj.num_mats
                assert(check(obj.x{k}), 'Data must be given as real nonempty arrays.');
                Nk = size(obj.x{k}, ndims(obj.x{k}));
                assert(N == Nk, 'All data arrays must have the same number of elements in their last dimension.');
            end
            
            % set remaining class properties
            obj.num_data = N;
            obj.nn = randperm(N);
            obj.i = 1;
        end
        
        % generates a new data batch
        [varargout] = gen(obj, N)
        
    end
end
