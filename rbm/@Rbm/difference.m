function [diff, stdev] = difference(obj, nade, type)
% Difference between this and a nade.
% INPUT
%     nade    nade to measure the difference to
%     type    type of difference measure
% OUTPUT
%     diff    difference estimate
%     stdev   standard deviation of the difference estimate
% 
% George Papamakarios, Jun 2015

assert(isa(nade, 'Nade') || isa(nade, 'NadeEnsemble'), 'Other model must be a nade or a nade ensemble.');

N = 10000;
burnin = 2000;
gibbs_state = obj.gibbs_state;
obj.setGibbsState(double(rand(obj.num_inputs, N) > 0.5));
[x, thisL] = obj.gen(N, burnin);
obj.setGibbsState(gibbs_state);

buf = 500;
othrL = zeros(1, N, nade.arraytype);
for i = 1:floor(N / buf)
    idx = (i-1)*buf + 1 : i*buf;
    othrL(idx) = nade.eval(x(:,idx));
end
rem = mod(N, buf);
othrL(end-rem+1:end) = nade.gen(x(:,end-rem+1:end));

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
