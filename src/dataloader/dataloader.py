import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io

class DiffusionDataset(Dataset):
    def __init__(self, file_path):
        # Load data from .mat file
        data = scipy.io.loadmat(file_path)
        
        # Extract variables and convert them to tensors
        x = torch.tensor(data['X'], dtype=torch.float32)  # Shape: (Nx, Nt)
        y = torch.tensor(data['Y'], dtype=torch.float32)  # Shape: (Nx, Nt)
        t = torch.tensor(data['t'], dtype=torch.float32)  # Shape: (Nx, Nt)
        u = torch.tensor(data['U'], dtype=torch.float32)  # Shape: (Nx, Nt)
        
        # Flatten the data to have samples of the form [x, t] with u as the target
        self.x = x.flatten()
        self.y = y.flatten()
        self.t = t.flatten()
        self.u = u.flatten() 

        # Create x and t for bcs and ics
        self.x_ic = x[0,:,0]
        self.y_ic = y[:,0,0]
        self.t_bc = t[0,0,:]
        
        # Load other parameters as constants
        self.dt = float(data['dt'])
        self.D = float(data['D'])
        
    def __len__(self):
        # Return the total number of samples
        return len(self.u)
    
    def __getitem__(self, idx):
        # Return a single sample
        return self.x[idx], self.y[idx], self.t[idx], self.u[idx]
    
    def getCondition(self):
        return self.x_ic, self.y_ic, self.t_bc


def create_dataloaders(file_path, batch_size=64, train_split=0.8, val_split=0.2):
    # Initialize dataset
    dataset = DiffusionDataset(file_path)
    
    # Calculate the number of samples for each split
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    # Ensure that the sum of train_size and val_size does not exceed the dataset_size
    train_size = min(train_size, dataset_size)
    val_size = min(val_size, dataset_size - train_size)
    
    # Split the dataset into train and validation sets
    train_set, val_set = random_split(dataset, [train_size, dataset_size - train_size])
    
    # The test set will be the entire dataset
    test_set = dataset
    
    # Create DataLoader for each set
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=dataset_size, shuffle=False)

    # Get the initial conditions and boundary conditions
    x_ic, y_ic, t_bc = dataset.getCondition()
    
    return train_loader, val_loader, test_loader, x_ic, y_ic, t_bc
