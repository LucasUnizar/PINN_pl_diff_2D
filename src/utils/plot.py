import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def plot_predictions_vs_true(predictions, true_values, gif_path="outputs/plots/pred_vs_gt_evolution_3D.gif"):
    Nx = 50
    Ny = 50
    Nt = 500
    
    # Convert tensors to numpy for plotting
    predictions = predictions.detach().cpu().numpy().reshape((Nx, Ny, Nt))
    true_values = true_values.detach().cpu().numpy().reshape((Nx, Ny, Nt))
    
    # Create a grid for x and y points
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    X, Y = np.meshgrid(y, x)  # Meshgrid to create a surface plot
    
    fig = plt.figure(figsize=(18, 8))
    ax_pred = fig.add_subplot(121, projection='3d')
    ax_true = fig.add_subplot(122, projection='3d')

    #Turn off interactive mode to speed up plotting
    plt.ioff()
    
    print('Creating 3D GIF...')

    def update(frame):
        # Clear previous plots
        ax_pred.clear()
        ax_true.clear()
        
        # Plotting predictions as a surface
        Z_pred = predictions[:, :, frame]
        ax_pred.plot_surface(X, Y, Z_pred, cmap='viridis')
        ax_pred.set_title(f'Predicted Values at Time Step {frame}')
        ax_pred.set_xlabel('Ny Points')
        ax_pred.set_ylabel('Nx Points')
        ax_pred.set_zlabel('Predicted Value')
        ax_pred.set_zlim(np.min(predictions), np.max(predictions))
        
        # Plotting true values as a surface
        Z_true = true_values[:, :, frame]
        ax_true.plot_surface(X, Y, Z_true, cmap='viridis')
        ax_true.set_title(f'True Values at Time Step {frame}')
        ax_true.set_xlabel('Ny Points')
        ax_true.set_ylabel('Nx Points')
        ax_true.set_zlabel('True Value')
        ax_true.set_zlim(np.min(true_values), np.max(true_values))
        
        return ax_pred, ax_true

    ani = animation.FuncAnimation(fig, update, frames=Nt, blit=False, repeat=False)

    # Save as GIF
    ani.save(gif_path, writer='pillow', fps=20)  # Adjust fps as needed
    plt.close(fig)

def plot_surf(variable, gif_path="outputs/plots/variable_evolution_3D.gif"):
    Nx = 50
    Ny = 50
    Nt = 500

    # Convert tensor to numpy array and reshape
    variable = variable.detach().cpu().numpy().reshape((Nx, Ny, Nt))

    # Create a grid for x and y points
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    X, Y = np.meshgrid(y, x)  # Meshgrid to create a surface plot

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Turn off interactive mode to speed up plotting
    plt.ioff()
    
    print('Creating 3D GIF...')

    def update(frame):
        # Clear previous plot
        ax.clear()
        
        # Plotting the variable as a surface
        Z = variable[:, :, frame]
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f'Values at Time Step {frame}')
        ax.set_xlabel('Ny Points')
        ax.set_ylabel('Nx Points')
        ax.set_zlabel('Value')
        ax.set_zlim(np.min(variable), np.max(variable))
        
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=Nt, blit=False, repeat=False)

    # Save as GIF
    ani.save(gif_path, writer='pillow', fps=20)  # Adjust fps as needed
    plt.close(fig)