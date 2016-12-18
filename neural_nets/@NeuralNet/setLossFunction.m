function setLossFunction(obj, loss_fun, lambda)
% Sets the network's loss function from a list of supported losses. Used
% for training the network.
% INPUTS
%       loss_fun     string identifier of the loss function
%       lambda       regularizer (needed only for combination of losses)
% 
% George Papamakarios, May 2015

% check input
assert(ischar(loss_fun), 'Loss function must be a string identifier.');

% set loss function
switch loss_fun
        
    case 'square_error'
        obj.loss_fun_hl = @obj.square_error;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = false;

    case 'cross_entropy'
        assert(obj.num_outputs == 1, 'Cross entropy works only with nets with a single output.');
        obj.loss_fun_hl = @obj.cross_entropy;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = false;

    case 'multi_cross_entropy'
        obj.loss_fun_hl = @obj.multi_cross_entropy;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = false;

    case 'dot_product'
        obj.loss_fun_hl = @obj.dot_product;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = false;

    case 'avg_score_matching'
        assert(obj.num_outputs == 1, 'Average score matching works only with nets with a single output.');
        obj.loss_fun_hl = @obj.avg_score_matching;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = true;

    case 'deriv_square_error'
        obj.loss_fun_hl = @obj.deriv_square_error;
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = true;

    case 'square_error_&_deriv_square_error'
        obj.loss_fun_hl = @(x, y) obj.square_error_and_deriv_square_error(x, y, lambda);
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = true;

    case 'cross_entropy_&_deriv_square_error'
        obj.loss_fun_hl = @(x, y) obj.cross_entropy_and_deriv_square_error(x, y, lambda);
        obj.loss_fun_id = loss_fun;
        obj.loss_fun_needs_derivs = true;
        
    otherwise
        error('Unsupported loss function.');
end
