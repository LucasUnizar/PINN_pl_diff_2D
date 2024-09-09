import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint callback
from src.dataloader.dataloader import create_dataloaders
from src.model.model import PINN  # Import the PINN class from your model
from src.model.base_models import MLP  # Import the MLP class from your model

class Solver:
    def __init__(self, args):
        self.args = args
        self._params(args)
        self._load_data()
        
        self.logger = WandbLogger(project=args.wandb_project, entity=args.wandb_entity)
        self.base_model = MLP(args.input_dim, args.hidden_dim, args.output_dim)
        self.model = PINN(self.base_model, lr=args.lr, x_ic=self.x_ic, y_ic=self.y_ic, t_bc=self.t_bc)

        self._load_trainer()
    
    def _params(self, args):
        self.bs = args.batch_size
        self.epochs = args.max_epochs
        self.path = 'data/diffusion_data_2D.mat'

    def _load_data(self):
        # Create the training data loader
        self.train_loader, self.valid_loader, self.test_loader, self.x_ic, self.y_ic, self.t_bc = create_dataloaders(self.path, batch_size=self.bs)
    
    def _load_trainer(self):
        # Define the checkpoint callback to save the model weights
        checkpoint_callback = ModelCheckpoint(
            dirpath='outputs/saved_models',  # Directory to save the model weights
            filename='best_model',  # Filename format
            save_top_k=1,  # Save only the best model
            monitor='val_loss',  # Metric to monitor for saving
            mode='min',  # Mode to select the best model (minimizing validation loss)
        )
        
        # Set up the PyTorch Lightning Trainer 
        self.trainer = pl.Trainer(
            num_sanity_val_steps=0,
            max_epochs=self.epochs,
            logger=self.logger,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",  
            callbacks=[checkpoint_callback],  
        )

    def train(self):
        # Train the model using the Trainer
        self.trainer.fit(self.model, self.train_loader, self.valid_loader)

    def test(self):
        print("Testing the model...")

        # Define the path to the best checkpoint file
        checkpoint_path = 'outputs/saved_models/best_model.ckpt'

        try:
            # Load the best checkpoint weights using the PINN class
            print(f"Loading weights from {checkpoint_path}")
            self.model = PINN.load_from_checkpoint(checkpoint_path, base_model=self.base_model, lr=self.args.lr, x_ic=self.x_ic, y_ic=self.y_ic, t_bc=self.t_bc)
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_path}. Testing with current model weights.")

        # Test the model using the Trainer
        self.trainer.test(self.model, self.test_loader)
