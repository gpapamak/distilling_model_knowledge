function [logZ, conf] = estimate_logZ(obj, nade, method, N)
% Estimates the log partition function of the rbm using sampling with nade
% as proposal distribution.
% INPUT
%     nade    nade to use as proposal distribution
%     method  sampling method to use
%     N       number of samples to use (optional)
% OUTPUT
%     logZ    estimate of log partition function
%     conf    confidence interval of logZ, conf = [lower, upper]
% 
% George Papamakarios, Jul 2015

assert(isa(nade, 'Nade'), 'Proposal distribution must be a nade.');
assert(obj.num_inputs == nade.num_inputs, 'Rbm and nade must have the same number of inputs.');

if nargin < 4
    N = 10000;
end

switch method
    
    case 'importance_sampling'
        [logZ, conf] = importance_sampling(obj, nade, N);
        
    case 'bridge_sampling'
        [logZ, conf] = bridge_sampling(obj, nade, N);
        
    otherwise
        error('Unknown method.');
end


function [logZ, conf] = importance_sampling(rbm, nade, N)
% Estimates logZ using importance sampling with N samples from nade.

% sample from nade
% note: sample in small batches to avoid running out of memory
buf = 500;
x = zeros(nade.num_inputs, N, nade.arraytype);
y = zeros(nade.num_inputs, N, nade.arraytype);
for i = 1:floor(N / buf)
    idx = (i-1)*buf + 1 : i*buf;
    [y(:,idx), x(:,idx)] = nade.gen(buf);
end
rem = mod(N, buf);
[y(:,end-rem+1:end), x(:,end-rem+1:end)] = nade.gen(rem);

% evaluate log probabilities of nade and rbm
Lnade = sum(x .* log(y) + (1-x) .* log(1-y), 1);
Lrbm = rbm.eval(x);

% estimate logZ
[logZ, conf] = compute_logR_estimate(Lrbm - Lnade);


function [logZ, conf] = bridge_sampling(rbm, nade, N)
% Estimates logZ using bridge sampling with N samples from rbm and nade.

buf = 500;
burnin = 2000;
epochs = 10;

% sample from nade and evaluate rbm and nade on the samples
% note: sample in small batches to avoid running out of memory
x2 = zeros(nade.num_inputs, N, nade.arraytype);
y2 = zeros(nade.num_inputs, N, nade.arraytype);
for i = 1:floor(N / buf)
    idx = (i-1)*buf + 1 : i*buf;
    [y2(:,idx), x2(:,idx)] = nade.gen(buf);
end
rem = mod(N, buf);
[y2(:,end-rem+1:end), x2(:,end-rem+1:end)] = nade.gen(rem);
L2nade = sum(x2 .* log(y2) + (1-x2) .* log(1-y2), 1);
L2rbm = rbm.eval(x2);

% sample from rbm and evaluate rbm and nade on the samples
% sampling the rbm is done with parallel chains to get independent samples,
% that are initialized with the samples from nade
% note: evaluate nade in small batches to avoid running out of memory
gibbs_state = rbm.gibbs_state;
rbm.setGibbsState(x2);
[x1, L1rbm] = rbm.gen(N, burnin);
rbm.setGibbsState(gibbs_state);
L1nade = zeros(1, N, nade.arraytype);
for i = 1:floor(N / buf)
    idx = (i-1)*buf + 1 : i*buf;
    L1nade(idx) = nade.eval(x1(:,idx));
end
rem = mod(N, buf);
L1nade(end-rem+1:end) = nade.eval(x1(:,end-rem+1:end));

% estimate logZ
% in each iteration refine bridge distribution given previous estimate
logZ = 0;
for i = 1:epochs
    L1star = bridge_distribution(L1rbm, L1nade, logZ);
    L2star = bridge_distribution(L2rbm, L2nade, logZ);
    [logR1, conf1] = compute_logR_estimate(L1star - L1rbm);
    [logR2, conf2] = compute_logR_estimate(L2star - L2nade);
    logZ = logR2 - logR1;
end
conf = conf2 - conf1([end,1]);


function [Lstar] = bridge_distribution(logP1, logP2, logR)
% Computes the log bridge distribution for bridge sampling.

Lstar = logP1 + logP2 - logsumexp([logP1; logR + logP2]);


function [logR, conf] = compute_logR_estimate(dL)
% Estimates logR = log(Z1 / Z2), given estimates dL = log(P1(x) / P2(x)),
% together with confidence intervals. Carefully takes care of numerical
% stability.

% estimate logR
dL = dL(:);
logN = log(length(dL));
logR = logsumexp(dL) - logN;

% estimate confidence interval of logR
numstd = 3;
logRsq = logsumexp(2*dL) - logN;
logstdR = (logRsq + log1p(-exp(min(2*logR - logRsq, 0))) - logN) / 2;
conf(1) = logR + log1p(-exp(min(logstdR - logR + log(numstd), 0)));
conf(2) = logR + log1p(numstd * exp(logstdR - logR));
