%% Burgers_test.m -- an executable m-file for solving a partial differential equation
% Automatically created in CHEBGUI by user Prateek Bhustali.
% Created on April 22, 2020 at 18:43.

%% Problem description.
% Solving
%   u_t = -u*(u)' + (0.01/pi)*u",
% for x in [-1,1] and t in [0,1], subject to
%   u = 0 at x = -1
% and
%   u = 0 at x = 1
tic
%% Problem set-up
% Create an interval of the space domain...
dom = [-1,1];
%...and specify a sampling of the time domain:
t = 0:.01:0.99;

% Make the right-hand side of the PDE.
pdefun = @(t,x,u) -u.*diff(u)+0.01./pi.*diff(u,2);

% Assign boundary conditions.
bc.left = 0;
bc.right = 0;

% Construct a chebfun of the space variable on the domain,
x = chebfun(@(x) x, dom);
% and of the initial condition.
u0 = -sin(8*pi*x);

%% Setup preferences for solving the problem.
opts = pdeset('Eps', 1e-7, 'HoldPlot', 'on', 'Ylim', [0,0.8]);

%% Call pde23t to solve the problem.
[t, u] = pde23t(pdefun, t, u0, bc, opts);

%% Plot the solution.
figure
% surf(u)
xlabel('x'), ylabel('t')
toc

x = (linspace(-1,1,256))';
time = 1:100;

usol(:,time) = u(x,time);
surf(usol)


filename = 'burgers_shock_IC_sin8pi.mat';

save(filename,'t','x','usol')
