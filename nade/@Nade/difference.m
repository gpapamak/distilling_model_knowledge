function [diff, stdev] = difference(obj, model, type)
% Difference between this nade and another model.
% INPUT
%     model   model to measure the difference to
%     type    type of difference measure
% OUTPUT
%     diff    difference estimate
%     stdev   standard deviation of the difference estimate
% 
% George Papamakarios, Jun 2015

assert(isa(model, 'Nade') || isa(model, 'NadeEnsemble') || isa(model, 'Rbm'), 'Other model must be a nade, a nade ensemble or an RBM.');

N = 10000;
buf = 500;
thisL = zeros(1, N, obj.arraytype);
othrL = zeros(1, N, model.arraytype);
for i = 1:floor(N / buf)
    idx = (i-1)*buf + 1 : i*buf;
    [y, x] = obj.gen(buf);
    thisL(idx) = sum(x .* log(y) + (1-x) .* log(1-y), 1);
    othrL(idx) = model.eval(x);
end
rem = mod(N, buf);
[y, x] = obj.gen(rem);
thisL(end-rem+1:end) = sum(x .* log(y) + (1-x) .* log(1-y), 1);
othrL(end-rem+1:end) = model.eval(x);

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
