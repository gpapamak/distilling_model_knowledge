function [m, S, epochs] = ep_gauss(x, y, prior_var, varargin)
% [m, S, epochs] = ep_gauss(x, y, prior_var, varargin)
% Expectation propagation for binary classification using gaussian factors.
% INPUT
%     x          DxN matrix with input points as columns
%     y          vector of N labels {-1,+1}
%     prior_var  variance of the spherical gaussian prior on the weights
%     -- optional name-value pairs -- 
%     method     specifies which binary classification function to use
%     maxepochs  run ep for at most that many passes over the data
%     tol        convergence tolerance
%     verbose    if true, print out info
%     eps        for step regression only, probability of the 'wrong' class
% OUTPUT
%     m         posterior mean
%     S         posterior covariance matrix
%     epochs    number of passes over the data till convergence
% 
% George Papamakarios, Jan 2015

p = inputParser;
p.addRequired('x', @(t) ismatrix(t) && isreal(t));
p.addRequired('y', @(t) numel(t) == size(x, 2) && all(t(:) == 1 | t(:) == -1));
p.addRequired('prior_var', @(t) isscalar(t) && t > 0);
p.addParameter('method', 'logistic', @(t) any(validatestring(t, {'step', 'probit', 'logistic'})));
p.addParameter('maxepochs', inf, @(t) isscalar(t) && t > 0 && (isint(t) || isinf(t)));
p.addParameter('tol', 1.0e-4, @(t) isscalar(t) && t >= 0);
p.addParameter('verbose', false, @(t) isscalar(t) && islogical(t));
p.addParameter('eps', 0.1, @(t) isscalar(t) && t >= 0);
p.parse(x, y, prior_var, varargin{:});

% prepare data
D = size(x, 1);
xy = x .* (ones(D, 1) * y(:)');

% initialize mean and covariance
m = zeros(D, 1);
S = prior_var * eye(D, D);

switch p.Results.method
    case 'step'
        msg_update = @(m, s) msg_update_step(m, s, p.Results.eps);
    case 'probit'
        msg_update = @msg_update_probit;
    case 'logistic'
        msg_update = @msg_update_logistic;
end

[m, S, epochs] = ep_rank_1(xy, m, S, msg_update, p.Results.maxepochs, p.Results.tol, p.Results.verbose);


function [m, S, epochs] = ep_rank_1(x, m, S, msg_update, maxepochs, tol, verbose)
% Expectation propagation in the case of rank 1 updates.

sqrt = @realsqrt;
N = size(x, 2);

% initialize
mn = zeros(N, 1);
sn = inf * ones(N, 1);
an = ones(N, 1);

epochs = 0;
diff = inf;
mn_prev = mn;
sn_prev = sn;
an_prev = an;

while diff > tol && epochs < maxepochs

    for n = randperm(N)
        
        % remove factor
        xSx = x(:,n)' * S * x(:,n); 
        d = 1 / (1 - xSx / sn(n));
        xSoldx = xSx * d;
        if xSoldx < 0
            % if removing the factor results in negative variance, choose a
            % different factor to remove
            continue;
        end
        xm = x(:,n)' * m;
        xmold = xm + xSoldx * (xm - mn(n)) / sn(n);

        % calculate messages
        [alpha, beta, Zn] = msg_update(xmold, xSoldx);

        % put factor back in
        m = m + (S * x(:,n)) * (d * ((xm - mn(n)) / sn(n) + alpha));
        Sx = S * x(:,n);
        S = S + (Sx * (d * (1/sn(n) - beta*d))) * Sx';
        S = (S + S') / 2;
        
        % update factor
        mn(n) = xmold + alpha / beta;
        sn(n) = 1/beta - xSoldx;
        an(n) = Zn * sqrt(1 + xSoldx/sn(n)) * exp(alpha^2 / (2*beta));
        
    end
    
    epochs = epochs + 1;
    diff_mn = abs(mn_prev - mn);
    diff_sn = abs(sn_prev - sn);
    diff_an = abs(an_prev - an);
    diff = max([diff_mn; diff_sn; diff_an]);
    mn_prev = mn;
    sn_prev = sn;
    an_prev = an;
    
    if verbose
        fprintf('Epoch = %d, difference = %g \n', epochs, diff);
    end
    
end


function [a, b, Z] = msg_update_step(m, s, e)
% Message update for the step factor.

t = m / sqrt(s);
Z = e + (1 - 2*e) * probit(t);
a = (1 - 2*e) * exp(log_gauss_pdf(t, 0, 1)) / (sqrt(s) * Z);
b = a * (a + m / s);


function [a, b, Z] = msg_update_probit(m, s)
% Message update for the probit factor.

t = m / sqrt(s + 1);
Z = probit(t);
a = exp(log_gauss_pdf(t, 0, 1)) / (sqrt(s + 1) * Z);
b = a * (a + m / (s + 1));


function [a, b, Z] = msg_update_logistic(m, s)
% Message update for the logistic factor. Uses MacKay's approximation to 
% the sigmoid-gaussian integral.

denom2 = 1 + (pi/8) * s;
Z = sigm(m / sqrt(denom2));
a = (1 - Z) / sqrt(denom2);
b = a * (a + (pi/8) * m) / denom2;
