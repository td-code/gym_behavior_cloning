import torch
import torch.nn as nn
import random
import time
from network import ClassificationNetwork
from utils import load_demonstrations

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False
import numpy as np

def train(data_file, trained_network_file, args):
    """
    Function for training the network.
    """
    infer_action = ClassificationNetwork()
    optimizer = torch.optim.Adam(infer_action.parameters(), lr=args.lr)

    # MODIFY CODE HERE
    # Loss function
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #wgt = 10*np.clip(1.0/np.array([813, 141, 7, 813, 141, 7, 1830, 1817, 80]), 0.01, 0.1)
    #wgt = torch.Tensor(wgt).to(device)
    #loss_function = nn.CrossEntropyLoss(weight=wgt)
    loss_function = nn.CrossEntropyLoss()

    observations, actions = load_demonstrations(data_file)
    observations = [torch.Tensor(observation) for observation in observations]
    actions = [torch.Tensor(action) for action in actions]

    batches = [batch for batch in zip(observations,
                                      infer_action.actions_to_classes(actions))]

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    number_of_classes = 9  # NEEDS TO BE ADJUSTED
    start_time = time.time()

    for epoch in range(nr_epochs):
        random.shuffle(batches)

        total_loss = 0
        batch_in = []
        batch_gt = []
        for batch_idx, batch in enumerate(batches):
            batch_in.append(batch[0].to(device))
            batch_gt.append(batch[1].to(device))

            if (batch_idx + 1) % batch_size == 0 or batch_idx == len(batches) - 1:
                batch_in = torch.reshape(torch.cat(batch_in, dim=0),
                                         (-1, 96, 96, 3))
                batch_gt = torch.reshape(torch.cat(batch_gt, dim=0), (-1, number_of_classes))

                batch_out = infer_action(batch_in)
                loss = loss_function(batch_out, batch_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss

                batch_in = []
                batch_gt = []

        time_per_epoch = (time.time() - start_time) / (epoch + 1)
        time_left = (1.0 * time_per_epoch) * (nr_epochs - 1 - epoch)
        print("Epoch %5d\t[Train]\tloss: %.6f \tETA: +%fs" % (
            epoch + 1, total_loss, time_left))

    torch.save(infer_action, trained_network_file)