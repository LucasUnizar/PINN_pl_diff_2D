clear all;
clc;

% Parameters
Lx = 10;             % Length of the spatial domain in x direction
Ly = 10;             % Length of the spatial domain in y direction
Nx = 50;             % Number of spatial points in x direction
Ny = 50;             % Number of spatial points in y direction
Dx = Lx / (Nx - 1);  % Spatial step size in x direction
Dy = Ly / (Ny - 1);  % Spatial step size in y direction
x = linspace(0, Lx, Nx); % Spatial grid in x
y = linspace(0, Ly, Ny); % Spatial grid in y

D = 10;              % Diffusion coefficient
dt = 0.001;          % Time step size
Nt = 500;            % Number of time steps
T = dt * Nt;         % Total simulation time

% Check CFL condition for stability
CFL_x = D * dt / Dx^2;
CFL_y = D * dt / Dy^2;
if CFL_x > 0.5 || CFL_y > 0.5
    error('The simulation is unstable. Reduce the time step dt or increase the spatial resolution Nx or Ny.');
end

% Initial conditions
[X, Y] = meshgrid(x, y);  % Create 2D grid
u0 = sin(pi * X / Lx) .* sin(pi * Y / Ly);  % Initial condition (example: sine wave in 2D)
u = u0;             % Initial value of u
u_new = u;          % Placeholder for updated values

% Replicate X and Y to match the third dimension (time)
X = repmat(X, 1, 1, Nt);  % X tensor of 50x50x500
Y = repmat(Y, 1, 1, Nt);  % Y tensor of 50x50x500

% Preallocate matrix to store solution over time
U = zeros(Nx, Ny, Nt + 1);
U(:, :, 1) = u;

% Time-stepping loop for diffusion
for t = 1:Nt
    % Update the solution using finite difference method in 2D
    u_new(2:end-1, 2:end-1) = u(2:end-1, 2:end-1) + D * dt * ( ...
        (u(3:end, 2:end-1) - 2 * u(2:end-1, 2:end-1) + u(1:end-2, 2:end-1)) / Dx^2 + ...
        (u(2:end-1, 3:end) - 2 * u(2:end-1, 2:end-1) + u(2:end-1, 1:end-2)) / Dy^2 );
    
    % Update the boundary conditions (assuming fixed boundary conditions)
    u_new(1, :) = 0;         % Boundary condition at y = 0
    u_new(end, :) = 0;       % Boundary condition at y = Ly
    u_new(:, 1) = 0;         % Boundary condition at x = 0
    u_new(:, end) = 0;       % Boundary condition at x = Lx
    
    % Update solution
    u = u_new;
    U(:, :, t + 1) = u;
end

% Create temporal vector
t_vec = linspace(0, T, Nt);  % Time vector

% Plotting the results
figure;
for t = 1:Nt
    surf(X(:, :, t), Y(:, :, t), U(:, :, t), 'EdgeColor', 'none');
    xlabel('X');
    ylabel('Y');
    zlabel('Concentration');
    title(['2D Diffusion at Time: ', num2str(t * dt)]);
    axis([0 Lx 0 Ly -0.1 1.1]);
    colorbar;
    pause(0.01);  % Pause for visualization
end

% Optional: Create a video
videoName = 'diffusion_simulation_2D.avi';
videoWriter = VideoWriter(videoName);
open(videoWriter);

for t = 1:Nt
    surf(X(:, :, t), Y(:, :, t), U(:, :, t), 'EdgeColor', 'none');
    xlabel('X');
    ylabel('Y');
    zlabel('Concentration');
    title(['Time: ', num2str(t * dt)]);
    axis([0 Lx 0 Ly -0.1 1.1]);
    drawnow;
    frame = getframe(gcf);
    writeVideo(videoWriter, frame);
end

close(videoWriter);
disp(['Video saved as ', videoName]);

% Postproc
U = U(:, :, 1:500);
t_vec = reshape(t_vec, 1, 1, Nt);  % Now t_vec is 1x1x500
t = repmat(t_vec, Nx, Ny, 1);  % Resulting tensor of 50x50x500

% Save the results to a .mat file
save('diffusion_data_2D.mat', 'X', 'Y', 't', 'U', 'dt', 'D');


