import argparse
import wandb

from src.solver.solver import Solver

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de PINN con PyTorch Lightning y Wandb")
    parser.add_argument('--train', action='store_true', help='Entrena el modelo')
    parser.add_argument('--input_dim', type=int, default=3, help='Dimensión de entrada')
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[32, 32], help='Dimensión de capa oculta')
    parser.add_argument('--output_dim', type=int, default=2, help='Dimensión de salida')
    parser.add_argument('--max_epochs', type=int, default=100, help='Número máximo de épocas')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamaño de lote')
    parser.add_argument('--lr', type=int, default=1.e-4, help='Tasa de aprendizaje')
    parser.add_argument('--wandb_project', type=str, default='PINN-Project', help='Nombre del proyecto en wandb')
    parser.add_argument('--wandb_entity', type=str, help='test')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    solver = Solver(args)
    if args.train:
        # Train the model
        solver.train()
    
    # Test the model
    solver.test()
    
if __name__ == '__main__':
    main()