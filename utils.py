import os
import torch
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, path, earlystop = 5, device="cuda"):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_acc = []
    train_acc = []
    min_loss = np.inf
    decrease = 1;
    not_improve = 0

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0       
        running_accuracy = 0 
        
        # training loop
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            # training phase            
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()  # reset gradient
            
            # forward
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)            

            # backward
            loss.backward()
            optimizer.step()  
                        
            running_loss += loss.item()
            running_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)

    
        model.eval()
        val_loss = 0
        val_accuracy = 0
        # validation loop
        with torch.no_grad():
            for _, data in enumerate(tqdm(val_loader)):                
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # evaluation metrics
                # loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)

        # calculate mean for each batch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(val_loss / len(val_loader))

        if min_loss > (val_loss / len(val_loader)):
            print('Loss Decreasing {:.3f} >> {:.3f} . Save model.'.format(min_loss, (val_loss / len(val_loader))))
            min_loss = (val_loss / len(val_loader))            
            torch.save(model.state_dict(), path)

        if ((val_loss / len(val_loader)) > min_loss):
            not_improve += 1
            min_loss = (val_loss / len(val_loader))
            print(f'Loss Not Decrease for {not_improve} time')
            if not_improve == earlystop:
                print(f'Loss not decrease for {not_improve} times, Stop Training')
                break

        train_acc.append(running_accuracy / len(train_loader))
        val_acc.append(val_accuracy / len(val_loader))
        print("Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
                "Train Acc:{:.3f}..".format(running_accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(val_accuracy / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60))

    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def test(model, test_loader, criterion, device="cuda"):
    model.eval()
    model.to(device)
    test_loss = 0
    test_accuracy = 0
    yPred = []
    y_test = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            inputs, labels = data        
            y_test.append(labels.detach().cpu().numpy()[0])            
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)                        
            yPred.append(preds.detach().cpu().numpy()[0])    
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_accuracy += torch.sum(preds == labels.data)/inputs.size(0)            
    
    print("Test Loss: {:.3f}".format(test_loss / len(test_loader)),
            "\nTest Accuracy: {:.3f}".format(test_accuracy / len(test_loader)))
    
    yPred = np.array(yPred)
    print('\n Classification report \n\n',
    classification_report(
        y_test,
        yPred,
        target_names=["B", "R", "RL", "L", "F"]
        )
    )

    print('\n Confusion matrix \n\n',
    confusion_matrix(
        y_test,
        yPred,
        )
    )

def plot_loss(history,path):
    plt.figure()
    plt.plot(history['val_loss'], label='val', marker='o')
    plt.plot(history['train_loss'], label='train', marker='o')
    plt.title('Loss per epoch');
    plt.ylabel('loss');
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    plt.savefig(os.path.join(path,'Loss.png'))
    
def plot_acc(history, path):
    plt.figure()
    plt.plot(history['train_acc'], label='train_accuracy', marker='*')
    plt.plot(history['val_acc'], label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch');
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()
    plt.savefig(os.path.join(path,'Accuracy.png'))
    
def save_plot(history, path):
    plot_loss(history, path)
    plot_acc(history, path)
