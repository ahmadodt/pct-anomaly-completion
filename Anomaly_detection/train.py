%load_ext autoreload
%autoreload 2

from data.dataset import AbnormalData
# import model
import argparse
import addict

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from data import ModelNet40
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import time 

def trainf(args, io):

    data , train_data, val_data, test_data = data_split()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    
    train_loader = DataLoader(
    train_data,
    num_workers=8,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    multiprocessing_context='spawn'
)
    val_loader = DataLoader(
    val_data,
    num_workers=8,
    batch_size=args.batch_size,
    shuffle=False,
    drop_last=True,
    multiprocessing_context='spawn'
)
    model = Pct(args,num_classes = len(data.classes),num_anomalies =len(data.anomalies) ).to(device)
    if args.saved:
        # Load the saved state dictionary
        checkpoint_path = 'checkpoints/%s/models/%s' % (args.exp_name, args.name)
        if os.path.exists(checkpoint_path):
            state_dict  = torch.load(checkpoint_path, map_location=torch.device('cuda'))  
            
            if 'module' in list(state_dict.keys())[0] and not hasattr(model, 'module'):
                # If the saved model was wrapped in DataParallel, but the current model is not
                state_dict = {k[7:]: v for k, v in state_dict.items()}  # Remove the 'module.' prefix
            model.load_state_dict(state_dict)
            # Load the state dictionary into the model
            #model.load_state_dict(checkpoint) 
        else: print("Checkpoint file does not exist. Model will be initialized with random weights.")
    
            
    # print(str(model))
    model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion_class = cal_loss
    criterion_anomaly = torch.nn.CrossEntropyLoss()
    best_test_acc_classes = 0
    best_test_acc_anomalies = 0
    #train arrays for visualization later
    train_losses = []
    train_accuracies_class = []
    train_balanced_accuracies_class = []
    train_accuracies_anomaly = []
    train_balanced_accuracies_anomaly = []
    # val arrays for visualization later
    val_losses = []
    val_accuracies_class = []
    val_balanced_accuracies_class = []
    val_accuracies_anomaly = []
    val_balanced_accuracies_anomaly = []

    #we need to add the validation part or we can skip that? --DONE
    #we need to add the testing part 
    # we need to maybe increase the dataset maybe doubling everyoneby dublicating and by using different points 
    #visualize watever you want nah 
    # visualize the train and test with time 
    # you can work on the overall loss maybe it could be easier for you to plot it 
    # we notice that the loss of the classification was okayish 
    # dont forget the code for saving the model 
    # make a test function alone better i think 
    #so plot val and train and then test and print the acccuracy this could be 2 pages of results.
    # talk about the other part whihc is the segmetation 
    for epoch in range(args.epochs):
        print(f'epoch {epoch} start:')
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred_class = []
        train_true_class = []
        train_pred_anomaly = []
        train_true_anomaly = []
        idx = 0
        total_time = 0.0
        for data, label_class, label_anomaly in (train_loader):
            data, label_class, label_anomaly = data.to(device), label_class.to(device).squeeze(), label_anomaly.to(device)
            #print("part one")

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            #print("part one")
            start_time = time.time()
            logits_class, logits_anomaly = model(data)
            
            #print("logits_class", logits_class[0])
            #print("label_class", label_class[0])
            
    
            loss_class = criterion_class(logits_class, label_class)
            
            loss_anomaly = criterion_anomaly(logits_anomaly, label_anomaly)
             #lossssssssssssssssssss
            loss = loss_class + 5*loss_anomaly
            ##########################

            loss.backward()
            #print("after loss backward")
            opt.step()
            #print("part one")
            scheduler.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            #print("part one")
            preds_class = logits_class.max(dim=1)[1]
            preds_anomaly = logits_anomaly.max(dim=1)[1]
            #print("part one")
            count += batch_size
            train_loss += loss.item() * batch_size
        
            train_true_class.append(label_class.cpu().numpy())
            train_pred_class.append(preds_class.detach().cpu().numpy())

            train_true_anomaly.append(label_anomaly.cpu().numpy())
            train_pred_anomaly.append(preds_anomaly.detach().cpu().numpy())
            idx += 1
            
        print ('train total time is',total_time)
        train_true_class = np.concatenate(train_true_class)
        train_pred_class = np.concatenate(train_pred_class)
        train_true_anomaly = np.concatenate(train_true_anomaly)
        train_pred_anomaly = np.concatenate(train_pred_anomaly)
        outstr = 'Train %d, loss: %.6f, train acc class: %.2f%%, train avg acc class: %.2f%%, train acc anomaly: %.2f%%, train avg acc anomaly: %.2f%%' % (epoch,
                                                                        train_loss * 1.0 / count,
                                                                        metrics.accuracy_score(train_true_class, train_pred_class) * 100,
                                                                        metrics.balanced_accuracy_score(train_true_class, train_pred_class) * 100,
                                                                        metrics.accuracy_score(train_true_anomaly, train_pred_anomaly) * 100,
                                                                        metrics.balanced_accuracy_score(train_true_anomaly, train_pred_anomaly) * 100
                                                                        )
        
        # Append metrics to lists for plotting
        train_losses.append(train_loss * 1.0 / count)
        train_accuracies_class.append(metrics.accuracy_score(train_true_class, train_pred_class))
        train_balanced_accuracies_class.append(metrics.balanced_accuracy_score(train_true_class, train_pred_class))
        train_accuracies_anomaly.append(metrics.accuracy_score(train_true_anomaly, train_pred_anomaly))
        train_balanced_accuracies_anomaly.append(metrics.balanced_accuracy_score(train_true_anomaly, train_pred_anomaly))

        io.cprint(outstr)
        
        
        # Validation Loop
        if (epoch % args.val_every_nepoch ==0):
            
            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_count = 0.0
            val_true_class = []
            val_pred_class = []
            val_true_anomaly = []
            val_pred_anomaly = []
            
            with torch.no_grad():  # Disable gradient computation during validation
                for val_data, val_label_class, val_label_anomaly in val_loader:
                    val_data, val_label_class, val_label_anomaly = val_data.to(device), val_label_class.to(device).squeeze(), val_label_anomaly.to(device)
                    val_data = val_data.permute(0, 2, 1)
                    
                    val_logits_class, val_logits_anomaly = model(val_data)
                    val_loss_class = criterion_class(val_logits_class, val_label_class)
                    val_loss_anomaly = criterion_anomaly(val_logits_anomaly, val_label_anomaly)
                    
                    #the losss summ
                    val_loss = val_loss_class + 5*val_loss_anomaly
                    # #############
                    
                    val_preds_class = val_logits_class.max(dim=1)[1]
                    val_preds_anomaly = val_logits_anomaly.max(dim=1)[1]
        
                    val_count += val_data.size(0)
                    val_loss += val_loss.item() * val_data.size(0)
        
                    val_true_class.append(val_label_class.cpu().numpy())
                    val_pred_class.append(val_preds_class.detach().cpu().numpy())
        
                    val_true_anomaly.append(val_label_anomaly.cpu().numpy())
                    val_pred_anomaly.append(val_preds_anomaly.detach().cpu().numpy())
        
            # Calculate and print validation metrics
            val_loss /= val_count
            val_true_class = np.concatenate(val_true_class)
            val_pred_class = np.concatenate(val_pred_class)
            val_true_anomaly = np.concatenate(val_true_anomaly)
            val_pred_anomaly = np.concatenate(val_pred_anomaly)
            
            val_outstr = 'Validation %d, loss: %.6f, val acc class: %.2f%%, val avg acc class: %.2f%%, val acc anomaly: %.2f%%, val avg acc anomaly: %.2f%%' % (epoch,
                                                                            val_loss,
                                                                            metrics.accuracy_score(val_true_class, val_pred_class) * 100,
                                                                            metrics.balanced_accuracy_score(val_true_class, val_pred_class) * 100,
                                                                            metrics.accuracy_score(val_true_anomaly, val_pred_anomaly) * 100,
                                                                            metrics.balanced_accuracy_score(val_true_anomaly, val_pred_anomaly) * 100
                                                                            )

            # Append metrics to lists for plotting
            val_losses.append(val_loss)
            val_accuracies_class.append(metrics.accuracy_score(val_true_class, val_pred_class))
            val_balanced_accuracies_class.append(metrics.balanced_accuracy_score(val_true_class, val_pred_class))
            val_accuracies_anomaly.append(metrics.accuracy_score(val_true_anomaly, val_pred_anomaly))
            val_balanced_accuracies_anomaly.append(metrics.balanced_accuracy_score(val_true_anomaly, val_pred_anomaly))
        
            io.cprint(val_outstr)
            # Ensure the directory exists before saving the model
            save_dir = 'checkpoints/%s/models/' % args.exp_name
            os.makedirs(save_dir, exist_ok=True)
            
            # Save the model
            torch.save(model.state_dict(), os.path.join(save_dir, '1kpoints_unbalancedLoss.pth'))

    train_accuracies = [train_losses, train_accuracies_class, train_balanced_accuracies_class, train_accuracies_anomaly, train_balanced_accuracies_anomaly]
    val_accuracies = [val_losses, val_accuracies_class, val_balanced_accuracies_class, val_accuracies_anomaly, val_balanced_accuracies_anomaly]
    #torch.save(model.state_dict(), 'workspace/pct-point-cloud-transformer-modified-main/checkpoints/%s/models/model.t7' % args.exp_name)
    
    return model,train_accuracies,val_accuracies


