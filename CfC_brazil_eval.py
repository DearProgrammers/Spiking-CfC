# Code created by Zhaojing Huang on 05 Deceomber 2023

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from ncps import wirings
from ncps.torch import CfC, LTC

import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen

from utils.filters import apply_bandpass_filter, filter_ecg_signal, resample_ecg_data, set_channels_to_zero, STFT_ECG_all_channels, min_max_normalize

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_fscore_support



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
parser.add_argument('--file_name', metavar="file_name", type=str, default='eval', help='Enter the model name you want to save as')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
# parser.add_argument('--num_repeats', type=int, default=0, help='Number of times to repeat the samples')
parser.add_argument('--n_channels', type=int, default=0, help='Number of channels are emptied')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--beta', type=float, default=0.4, help='Spiking beta value')
parser.add_argument('--num_steps', type=int, default=9, help='Number of steps')
parser.add_argument('--threshold', type=float, default=0.17, help='Threshold for spike')
parser.add_argument('--pov_rate', type=int, default=6, help='Positive vs negative label ratio')
parser.add_argument('--gpu', type=int, default=0, help='The GPU this will run on')




# Input
args = parser.parse_args()

file_name = args.file_name
epochs = args.epochs
batch_size = args.batch_size
# num_repeats = args.num_repeats
n = args.n_channels
lr = args.learning_rate
dr = args.dropout_rate
system = str(args.gpu)

# Spike settings
beta = args.beta  # neuron decay rate
num_steps = args.num_steps
threshold = args.threshold

pov_rate = args.pov_rate

slope = 10
kernel_size = 9


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
x_val = np.load('/x_val.npy')[: , 1, :, :].reshape(-1, 1, 129, 33).astype(np.float32)
y_val = np.load('/y_val.npy')
label = ['1dAVb','RBBB','LBBB','SB','AF','ST']



print(x_val.shape, y_val.shape)

# Train
SNN_net = Net(dr, spike_grad, threshold, beta, kernel_size).to(device)

state_dict = torch.load('/model.pth')

# Load the state dictionary into the model
SNN_net.load_state_dict(state_dict)

# Perform inference in batches
all_probs = []  # List to store probabilities for all batches

for start_idx in range(0, len(x_val), batch_size):
    end_idx = start_idx + batch_size
    batch_x_val = x_val[start_idx:end_idx]

    # Convert batch_x_val to a PyTorch tensor and move to the device
    batch_x_val_tensor = torch.tensor(batch_x_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        normalized_data = min_max_normalize(batch_x_val_tensor)
        processed_data = spikegen.rate(normalized_data, num_steps=num_steps)
        spk_rec, _ = SNN_net(processed_data)
        outputs = torch.sigmoid(spk_rec.sum(0))


    # Convert outputs to probabilities for the positive class
    all_probs.append(outputs.cpu().numpy())  # Store batch probabilities
    all_probs_array = np.concatenate(all_probs, axis=0)
    print(all_probs_array.shape)

# Initialize lists to store per-class metrics
class_metrics = []

for class_index, class_label in enumerate(label):
    y_val_binary = y_val[:, class_index]
    predictions = (all_probs_array[:, class_index] > 0.5).astype(int)

    accuracy = accuracy_score(y_val_binary, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val_binary, predictions, average='binary')
    auroc = roc_auc_score(y_val_binary, all_probs_array[:, class_index])

    class_metrics.append({
        'Class': class_label,
        'Recall': recall,
        'Precision': precision,
        'F1': f1,
        'AUROC': auroc,
    })

# Calculate average metrics
average_metrics = {
    'Class': 'Average',
    'Recall': np.mean([metric['Recall'] for metric in class_metrics]),
    'Precision': np.mean([metric['Precision'] for metric in class_metrics]),
    'F1': np.mean([metric['F1'] for metric in class_metrics]),
    'AUROC': np.mean([metric['AUROC'] for metric in class_metrics]),
}

# Append average metrics to the list of class metrics
class_metrics.append(average_metrics)

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame(class_metrics)

# Save the DataFrame to a CSV file
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Example usage:
create_directory('/model'+ file_name)

metrics_df.to_csv('/model/'+ file_name +'/results.csv', index=False)
