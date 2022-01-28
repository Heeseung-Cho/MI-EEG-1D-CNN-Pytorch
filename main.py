"""
A 1D CNN for high accuracy classiﬁcation in motor imagery EEG-based brain-computer interface
Journal of Neural Engineering (https://doi.org/10.1088/1741-2552/ac4430)
Copyright (C) 2022  Francesco Mattioli, Gianluca Baldassarre, Camillo Porcaro

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import argparse
import os
from models import HopefullNet
from dataloader import MIEEGdataloader
from utils import fit, test, save_plot
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser(description='MI-EEG-1D-CNN for Pytorch')

parser.add_argument('--epochs', default=100, type=int,
                    help='Epochs')
parser.add_argument('--batch_size', default=10, type=int,
                    help='Batch Size')
parser.add_argument('--earlystop', default=10, type=int,
                    help='Early Stopping')
parser.add_argument('--channel', default="a", type=str,
                    help='Channel has a,b,c,d,e,f')
parser.add_argument('--sourcepath', default="dataset/paper", type=str,
                    help='Set source path')
parser.add_argument('--savepath', default="saved_model", type=str,
                    help='Set save path')
parser.add_argument('--augment', default=True, type=int,
                    help='Set augmentation (SMOTE), 0: False, else: True')
parser.add_argument('--random_state', default=42, type=int,
                    help='Set augmentation parameter : Random state')
parser.add_argument('--k_neighbors', default=5, type=int,
                    help='Set augmentation : k_neighbors')
parser.set_defaults(argument=True)


def main():
    global args
    args = parser.parse_args()    

    # Set GPU number
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')

    #Params
    root_path = os.getcwd()
    source_path = os.path.join(root_path, args.sourcepath)
    save_name = "Augment"
    augment = True
    if args.augment == 0:
        save_name = "No_augment"    
        augment = False
    save_path = os.path.join(root_path, "saved_model", args.savepath)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_loader, valid_loader, test_loader = MIEEGdataloader(path = source_path,
                                                            channel = args.channel,
                                                            batch_size  = args.batch_size,
                                                            augment  = augment,
                                                            random_state = args.random_state,
                                                            k_neighbors= args.k_neighbors)    
    learning_rate = 1e-4

    model = HopefullNet()
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)    
    modelPath = os.path.join(save_path, f'bestModel_{save_name}.pt')
    history = fit(args.epochs, model, train_loader, valid_loader, loss, optimizer, modelPath, args.earlystop, device = device)

    with open(os.path.join(save_path, f"hist_{save_name}.pkl"), "wb") as file:
        pickle.dump(history, file)

    """
    Test model
    """
    del model
    model = HopefullNet()
    model.load_state_dict(torch.load(modelPath))

    test(model, test_loader, loss, device)
    #save_plot(history, save_path)
    
    
if __name__ == "__main__":
    main()