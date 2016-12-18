% ADF and EP demo for logistic regression.
% George Papamakarios, Jan 2015

clear;
close all;

npoints = 100;
x1 = randn(2, npoints) + 2*ones(2,npoints);
x2 = randn(2, npoints) - 2*ones(2,npoints);
y1 = +ones(1,npoints);
y2 = -ones(1,npoints);
x = [x1 x2];
y = [y1 y2];

tau = 100;
%[m, S, Z] = adf_logreg(x, y, tau);
[m, S, epochs] = ep_gauss(x, y, tau, 'verbose', true, 'method', 'logistic');

figure; hold on;
plot(x1(1,:), x1(2,:), 'bo');
plot(x2(1,:), x2(2,:), 'rx');

xx = [min(x(1,:)) max(x(1,:))] + 2*[-1 1];
yy = [min(x(2,:)) max(x(2,:))] + 2*[-1 1];
nws = 100;
ws = gauss_sample(m, S, nws);
for i = 1:nws
    plot(xx, -ws(1,i)/ws(2,i)*xx, 'g');
end
plot(xx, -m(1)/m(2)*xx, 'k', 'LineWidth', 2);

axis equal;
xlim(xx);
ylim(yy);
