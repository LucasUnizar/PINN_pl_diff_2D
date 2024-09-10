import numpy as np
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.utils.plot import plot_predictions_vs_true, plot_surf

class PINN(pl.LightningModule):
    def __init__(self, base_model, lr, x_ic, y_ic, t_bc, criterion=torch.nn.MSELoss(), optimizer=torch.optim.Adam):
        super().__init__()
        self.model = base_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.x_ic, self.y_ic, self.t_bc = x_ic, y_ic, t_bc

        # Containers for storing losses
        self.train_losses = []
        self.val_losses = []

    def forward(self, features):
        # Forward pass through the model
        return self.model(features)

    def data_driven_loss(self, x, y, t, u):
        # Stack (x, y, t) for input
        features = torch.stack([x, y, t], dim=1)
        # Calculate data-driven loss between predictions and ground truth data
        predictions, _ = torch.unbind(self.model(features), dim=1)
        loss = self.criterion(predictions, u)
        return loss

    def physics_loss(self, x, y, t):
        # Stack (x, y, t) for input
        features = torch.stack([x, y, t], dim=1)
        # Obtain predictions from the model
        u, alpha = torch.unbind(self.model(features), dim=1)  # Model output includes u, and alpha
        
        # Compute gradients with respect to x, y, and t
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        # Compute the second derivatives w.r.t. x and y
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        
        # Calculate the PDE residual
        residual = u_t - alpha * (u_xx + u_yy)
        
        # Compute the loss (L2 norm of the residual)
        loss = torch.mean(residual**2)
        
        return loss, alpha.mean()

    def ic_loss(self, x, y):
        """Calculates the initial condition loss."""
        # Condition at t = 0
        t = torch.zeros_like(x)
        features = torch.stack([x, y, t], dim=1)

        # Predict using the model
        u_pred, _ = torch.unbind(self.model(features), dim=1)

        # Initial condition u0 = sin(pi * x / L) * sin(pi * y / L)
        L = 10
        u0 = torch.sin(np.pi * x / L) * torch.sin(np.pi * y / L)

        # MSE Loss between the prediction and the initial condition
        loss = torch.mean((u_pred - u0) ** 2)
        return loss

    def bc_loss(self, x, y, t):
        """Calculates the boundary condition loss."""
        # Interpol dim
        x = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=len(t), mode='linear', align_corners=True).squeeze(0).squeeze(0).to(self.device)
        y = F.interpolate(y.unsqueeze(0).unsqueeze(0), size=len(t), mode='linear', align_corners=True).squeeze(0).squeeze(0).to(self.device)

        # Condition at boundaries: x = 0, x = 10, y = 0, y = 10
        x0 = torch.zeros_like(t)  # x = 0
        x10 = torch.ones_like(t) * 10  # x = 10
        y0 = torch.zeros_like(t)  # y = 0
        y10 = torch.ones_like(t) * 10  # y = 10

        # Stack (x, y, t) for all boundaries
        features_x0 = torch.stack([x0, y, t], dim=1)
        features_x10 = torch.stack([x10, y, t], dim=1)
        features_y0 = torch.stack([x, y0, t], dim=1)
        features_y10 = torch.stack([x, y10, t], dim=1)

        # Predict using the model
        u_x0_pred, _ = torch.unbind(self.model(features_x0), dim=1)
        u_x10_pred, _ = torch.unbind(self.model(features_x10), dim=1)
        u_y0_pred, _ = torch.unbind(self.model(features_y0), dim=1)
        u_y10_pred, _ = torch.unbind(self.model(features_y10), dim=1)

        # Boundary condition u = 0 at all boundaries
        loss_x0 = torch.mean(u_x0_pred ** 2)  # u(0, y, t) = 0
        loss_x10 = torch.mean(u_x10_pred ** 2)  # u(10, y, t) = 0
        loss_y0 = torch.mean(u_y0_pred ** 2)  # u(x, 0, t) = 0
        loss_y10 = torch.mean(u_y10_pred ** 2)  # u(x, 10, t) = 0

        # Total boundary loss
        loss = loss_x0 + loss_x10 + loss_y0 + loss_y10
        return loss

    def training_step(self, batch, batch_idx):
        # Unpack batch data
        x, y, t, u = batch
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        # Calculate different components of the loss
        data_loss = self.data_driven_loss(x, y, t, u)
        physics_loss, _ = self.physics_loss(x, y, t)
        ic_loss = self.ic_loss(self.x_ic.to(self.device), self.y_ic.to(self.device))
        bc_loss = self.bc_loss(self.x_ic, self.y_ic, self.t_bc.to(self.device))

        # Combine all loss components
        total_loss = data_loss + physics_loss + ic_loss + bc_loss

        # Log losses for monitoring
        self.log("train_loss", total_loss)
        self.log("train_data_loss", data_loss)
        self.log("train_physics_loss", physics_loss)
        self.log("train_ic_loss", ic_loss)
        self.log("train_bc_loss", bc_loss)

        # Append losses to list for tracking
        self.train_losses.append({
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item(),
            "ic_loss": ic_loss.item(),
            "bc_loss": bc_loss.item()
        })
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            # Unpack batch data
            x, y, t, u = batch
            x.requires_grad = True
            y.requires_grad = True
            t.requires_grad = True

            # Calculate different components of the loss
            data_loss = self.data_driven_loss(x, y, t, u)
            physics_loss, alpha = self.physics_loss(x, y, t)
            ic_loss = self.ic_loss(self.x_ic.to(self.device), self.y_ic.to(self.device))
            bc_loss = self.bc_loss(x, y, self.t_bc.to(self.device))

            # Combine all loss components
            total_loss = data_loss + physics_loss + ic_loss + bc_loss

            # Log losses for monitoring
            self.log("val_loss", total_loss)
            self.log("val_data_loss", data_loss)
            self.log("val_physics_loss", physics_loss)
            self.log("val_ic_loss", ic_loss)
            self.log("val_bc_loss", bc_loss)
            self.log("val_alpha", alpha.mean())

            # Append losses to list for tracking
            self.val_losses.append({
                "total_loss": total_loss.item(),
                "data_loss": data_loss.item(),
                "physics_loss": physics_loss.item(),
                "ic_loss": ic_loss.item(),
                "bc_loss": bc_loss.item(),
                "alpha": alpha.mean()
            })

    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            # Unpack batch data
            x, y, t, u = batch
            x.requires_grad = True
            y.requires_grad = True
            t.requires_grad = True
            # Stack (x, y, t) for input
            features = torch.stack([x, y, t], dim=1)
            # Calculate data-driven loss between predictions and ground truth data
            predictions, alpha = torch.unbind(self.model(features), dim=1)
            # Collapse Alpha to a scalar
            alpha_mean = alpha.mean()
            
            # Print alpha value
            self.log("Alpha_inference_value", alpha_mean.item())

            # Plot predictions vs true
            plot_predictions_vs_true(predictions, u)
            plot_surf(alpha, gif_path="outputs/plots/alpha_evolution_3D.gif")
            error = torch.sqrt((predictions - u)**2)
            plot_surf(error, gif_path="outputs/plots/qerror_evolution_3D.gif")


    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = self.optimizer(self.parameters(), lr=self.lr)

        # Set up the learning rate scheduler - ReduceLROnPlateau
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=10, 
                threshold=0.0001, 
                min_lr=1e-6, 
                verbose=True
            ),
            'monitor': 'val_loss',  # Monitor the validation loss
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
