# Code created by Zhaojing Huang for training the model on 05 Deceomber 2023

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from ncps import wirings
from ncps.torch import CfC, LTC

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen

from utils.filters import apply_bandpass_filter, filter_ecg_signal, resample_ecg_data, set_channels_to_zero, STFT_ECG_all_channels

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score



import pandas as pd
import h5py
import numpy as np
import os
import csv
import pickle
import argparse




parser = argparse.ArgumentParser(
    prog='Model Name',
    description='What do you want to save your Model as',
    epilog='Name of the model'
)
parser.add_argument('--file_name', metavar="file_name", type=str, default='test', help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--num_repeats', type=int, default=0, help='Number of times to repeat the samples')
parser.add_argument('--n_channels', type=int, default=0, help='Number of channels are emptied')
parser.add_argument('--learning_rate', type=float, default=0.00015, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.06, help='Dropout rate')
parser.add_argument('--beta', type=float, default=0.4, help='Spiking beta value')
parser.add_argument('--num_steps', type=int, default=10, help='Number of steps')
parser.add_argument('--threshold', type=float, default=0.56, help='Threshold for spike')
parser.add_argument('--pov_rate', type=int, default=6, help='Positive vs negative label ratio')
parser.add_argument('--gpu', type=int, default=0, help='The GPU this will run on')
parser.add_argument('--slope', type=int, default=10, help='The slope of gradient')
parser.add_argument('--kernel_size', type=int, default=9, help='The kernel size')




# Input
args = parser.parse_args()

file_name = args.file_name
epochs = args.epochs
batch_size = args.batch_size
num_repeats = args.num_repeats
n = args.n_channels
lr = args.learning_rate
dr = args.dropout_rate
system = str(args.gpu)

# Spike settings
beta = args.beta  # neuron decay rate
num_steps = args.num_steps
threshold = args.threshold

pov_rate = args.pov_rate

slope = args.slope
kernel_size = args.kernel_size


spike_grad = surrogate.fast_sigmoid(slope=slope)  # surrogate gradient
spike_grad_lstm = surrogate.straight_through_estimator()


# Device setting
os.environ['CUDA_VISIBLE_DEVICES'] = system
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# The Model
wiring = wirings.AutoNCP(20, 6)
ncp = CfC(75, wiring, batch_first=True) #, return_sequences=False

#
# PyTorch model in Class
class Net(nn.Module):
    def __init__(self, dr, spike_grad, threshold, beta, kernel_size):
        super(Net, self).__init__()
        # input shape None, 12, 129, 33
        self.lstm1 = snn.SConv2dLSTM(in_channels=1, out_channels=16, kernel_size=kernel_size, max_pool=2, threshold=threshold, spike_grad=spike_grad_lstm)
        self.fc1 = nn.Linear(16384, 75)
        self.dropout1 = nn.Dropout(dr)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)
        self.ncp = ncp
        # self.sigmoid = nn.Sigmoid()
        self.lif2 =  snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

    def forward(self, x):
        # Initialize LIF state variables and spike output tensors
        syn1, mem1 = self.lstm1.init_sconv2dlstm()
        mem2 = self.lif1.init_leaky()
        mem3 = self.lif2.init_leaky()
        spk3_rec = []
        mem3_rec = []


        for step in range(x.size(0)):
            spk1, syn1, mem1 = self.lstm1(x[step], syn1, mem1) # .unsqueeze(1)
            cur2 = self.dropout1(self.fc1(spk1.flatten(1)))
            spk2, mem2 = self.lif1(cur2, mem2)
            cur3, _ = ncp(spk2)
            spk3, mem3 = self.lif2(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)




# Data section
print('Reading Data')
# Reading Y
path_to_csv = '//'
label = pd.read_csv(path_to_csv)[['1dAVb','RBBB','LBBB','SB','AF','ST']]
columns = label.columns
# Convert label values to np.float32 data type
y = label.values.astype(np.float32)


# Reading X
path_to_hdf5 = '//'
hdf5_dset = 'tracings'
f = h5py.File(path_to_hdf5, "r")
x = f[hdf5_dset][:]

print('Resampling X')
x = resample_ecg_data(x, 400, 500, 4096)
print('Band passing X')
x = apply_bandpass_filter(x)
print('Filtering X')
x = filter_ecg_signal(x)
print('Emptying X channels')
x = set_channels_to_zero(x, n)

print('Transforming x')
x = STFT_ECG_all_channels(500, x)
print(x.shape)


print(x.shape, y.shape)


# Data normalisation
def min_max_normalize(x):
    x_array = x.cpu().numpy()  # Convert tensor to NumPy array on CPU

    nonzero_indices = np.any(x_array != 0, axis=(1, 2, 3))

    if np.any(nonzero_indices):
        x_nonzero = x_array[nonzero_indices]
        min_values = np.min(x_nonzero, axis=(1, 2, 3), keepdims=True)
        max_values = np.max(x_nonzero, axis=(1, 2, 3), keepdims=True)

        normalized_x_nonzero = (x_nonzero - min_values) / (max_values - min_values)

        normalized_x_array = np.zeros_like(x_array)
        normalized_x_array[nonzero_indices] = normalized_x_nonzero

        normalized_x_tensor = torch.tensor(normalized_x_array, dtype=x.dtype,
                                           device=x.device)  # Convert array back to tensor on the same device
        return normalized_x_tensor


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42, shuffle=True)


# Resampling Data
indices = np.where(y_train.sum(axis=1) > 0)[0]
sampled_y_train = y_train[indices]
sampled_x_train = x_train[indices]

# Repeat the samples
repeated_y_train = np.repeat(sampled_y_train, num_repeats, axis=0)
repeated_x_train = np.repeat(sampled_x_train, num_repeats, axis=0)

# Concatenate the repeated samples with the original training dataset
y_train = np.concatenate((y_train, repeated_y_train), axis=0)
x_train = np.concatenate((x_train, repeated_x_train), axis=0)

print(x_train.shape, y_train.shape, type(x_train), type(y_train))
print(x_val.shape, y_val.shape, type(x_val), type(y_val))



# Convert x_val and y_val to numpy arrays if they are not already
x_val = np.array(x_val)
y_val = np.array(y_val)


# Convert NumPy arrays to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# Create a TensorDataset
dataset = TensorDataset(x_train_tensor, y_train_tensor)
# Define batch size and create a DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Convert NumPy arrays to PyTorch tensors
x_test_tensor = torch.tensor(x_val, dtype=torch.float32)
y_test_tensor = torch.tensor(y_val, dtype=torch.float32)
# Create a TensorDataset
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
# Create a DataLoader for testing data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)






# Train
SNN_net = Net(dr, spike_grad, threshold, beta, kernel_size).to(device)

# Define class weights
class_weights = torch.tensor([pov_rate], dtype=torch.float32, device=device)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
optimizer = torch.optim.AdamW(SNN_net.parameters(), lr=lr, betas=(0.9, 0.999))
epochs = epochs
loss_hist = []
val_loss_hist = []
counter = 0

train_csv_path = 'log/' + file_name + '/train_performance.csv'
val_csv_path = 'log/' + file_name + '/val_performance.csv'
train_csv_header = ['Epoch', 'TrainLoss', 'TrainAccuracy', 'TrainPrecision', 'TrainRecall', 'TrainF1', 'TrainAUROC', 'TrainAUPRC']
val_csv_header = ['Epoch', 'ValidationLoss', 'ValAccuracy', 'ValPrecision', 'ValRecall', 'ValF1', 'ValAUROC', 'ValAUPRC']

os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
# Save x_val and y_val as numpy arrays
np.save('/mnt/data13_16T/jim/ECG/Codes/NCP-PyTorch/log/' + file_name + '/x_val.npy', x_val)
np.save('/mnt/data13_16T/jim/ECG/Codes/NCP-PyTorch/log/' + file_name + '/y_val.npy', y_val)



# Save argument values to a text file
with open('log/' + file_name + '/argument_values.txt', 'w') as f:
    f.write(f'File Name: {args.file_name}\n')
    f.write(f'Epochs: {args.epochs}\n')
    f.write(f'Learning Rate: {args.learning_rate}\n')
    f.write(f'Dropout Rate: {args.dropout_rate}\n')
    f.write(f'Batch Size: {args.batch_size}\n')
    f.write(f'Num Repeats: {args.num_repeats}\n')
    f.write(f'Num Channels: {args.n_channels}\n')
    f.write(f'Beta: {args.beta}\n')
    f.write(f'Num Steps: {args.num_steps}\n')
    f.write(f'Threshold: {args.threshold}\n')
    f.write(f'P vs N rate: {args.pov_rate}\n')
    f.write(f'Slope: {slope}\n')
    f.write(f'Kernel: {kernel_size}\n')

print('Argument values saved to argument_values.txt')

with open(train_csv_path, mode='w', newline='') as train_csv_file, \
     open(val_csv_path, mode='w', newline='') as val_csv_file:
    train_csv_writer = csv.writer(train_csv_file)
    val_csv_writer = csv.writer(val_csv_file)
    train_csv_writer.writerow(train_csv_header)
    val_csv_writer.writerow(val_csv_header)


# Outer training loop
for epoch in range(epochs):

    # Minibatch training loop
    for input_data, targets in train_loader:
        normalized_data = min_max_normalize(input_data)
        processed_data = spikegen.rate(normalized_data, num_steps=num_steps)
        processed_data = processed_data.to(device)
        targets = targets.to(device)

        # forward pass
        SNN_net.train()
        spk_rec, _ = SNN_net(processed_data)

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=torch.float, device=device)
        loss_val += loss_fn(spk_rec.sum(0), targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Print train/test loss/accuracy
        if counter % 1 == 0:
            print(f"Epoch: {epoch+1}/{epochs}, Iteration: {counter}, Train Loss: {loss_val.item()}")

            with torch.no_grad():
                SNN_net.eval()  # Set the model to evaluation mode
                spk_rec, _ = SNN_net(processed_data)
                predictions = torch.sigmoid(spk_rec.sum(0))

                # Convert predictions to binary values (0 or 1)
                binary_predictions = (predictions > 0.5).int()

                accuracy = accuracy_score(targets.cpu().numpy().flatten(), binary_predictions.cpu().numpy().flatten())
                precision = precision_score(targets.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')
                recall = recall_score(targets.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')
                f1 = f1_score(targets.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')

                # AUROC and AUPRC need different handling for multi-class
                try:
                    auroc = roc_auc_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='macro',
                                      multi_class='ovr')
                except ValueError as e:
                    if "Only one class present in y_true" in str(e):
                        # Handle the case of only one class (e.g., print a message, set roc_auc to a default value)
                        roc_auc = 0.5  # Set a default value or any value that makes sense in your context
                    else:
                        # Re-raise the exception if it's a different ValueError
                        raise

                auprc = average_precision_score(targets.cpu().numpy(), predictions.cpu().numpy(), average='micro')

                with open(train_csv_path, mode='a', newline='') as train_csv_file:
                    train_csv_writer = csv.writer(train_csv_file)
                    train_csv_writer.writerow(
                        [counter + 1, loss_val.item(), accuracy, precision, recall, f1, auroc, auprc])

                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")
                print("---------------------------------------------------------------------------------")

        counter += 1


    with torch.no_grad():
        validation_loss = 0.0
        all_predictions = []
        all_targets = []

        for val_data, val_targets in iter(test_loader):
            val_data = min_max_normalize(val_data)
            val_data = spikegen.rate(val_data, num_steps=num_steps)
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)

            SNN_net.eval()  # Set the model to evaluation mode
            val_spk_rec, _ = SNN_net(val_data)
            val_loss = loss_fn(val_spk_rec.sum(0), val_targets)
            validation_loss += val_loss.item()

            val_predictions = torch.sigmoid(val_spk_rec.sum(0))
            all_predictions.append(val_predictions.cpu().numpy())
            all_targets.append(val_targets.cpu().numpy())

        validation_loss /= len(test_loader)
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)

        val_accuracy = accuracy_score(all_targets.flatten() > 0.5, all_predictions.flatten() > 0.5)
        val_precision = precision_score(all_targets > 0.5, all_predictions > 0.5, average='micro')
        val_recall = recall_score(all_targets > 0.5, all_predictions > 0.5, average='micro')
        val_f1 = f1_score(all_targets > 0.5, all_predictions > 0.5, average='micro')
        val_auroc = roc_auc_score(all_targets, all_predictions, average='macro', multi_class='ovr')
        val_auprc = average_precision_score(all_targets, all_predictions, average='micro')

        print("---------------------------------------------------------------------------------")
        print(f"Epoch: {epoch + 1}/{epochs} \t Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}, AUROC: {val_auroc:.4f}, AUPRC: {val_auprc:.4f}")

        checkpoint_path = f'log/{file_name}/model_checkpoint_epoch_{epoch+1}.pth'
        torch.save(SNN_net.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved: {checkpoint_path}")

        # Write the training and validation performance to CSV files
        with open(val_csv_path, mode='a', newline='') as val_csv_file:
            val_csv_writer = csv.writer(val_csv_file)
            val_csv_writer.writerow(
                [epoch + 1, validation_loss, val_accuracy, val_precision, val_recall, val_f1, val_auroc, val_auprc])

        print("---------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------")
