import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_model(model, test_loader, device, Loss):
    model.eval()
    total_loss = 0
    batch_count = 0
    true_labels = []
    pred_labels = []
    start_time = time.time()
    
    for batch in test_loader:
        text, video, audio, label = batch
       
        text, video, audio, label = text.to(device), audio.to(device), video.to(device), label.to(device)
        
        predit = model(text,  video, audio)
        _, preds = torch.max(predit, 1)
        label = F.one_hot(label.long().squeeze(), num_classes=7)
        true_labels.extend(label.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())
        
        loss = Loss(predit, label.squeeze())
        
        total_loss += loss.item()
        batch_count += 1

        return total_loss/batch_count, time.time()-start_time
    
def train_model(model, train_loader, test_loader, device, Loss, optimizer, epochs):
    model = model.to(device)
    for e in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        start_time = time.time()
        for batch in train_loader:
            text, video, audio, label = batch
            text, video, audio, label = text.to(device), audio.to(device), video.to(device), label.to(device)
            optimizer.zero_grad()
            predit = model(text,  video, audio)
            label = F.one_hot(label.long().squeeze(), num_classes=7)
            loss = Loss(predit, label.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        print('epoch :',e, 'loss :', total_loss/batch_count, 'time :', time.time()-start_time)
        test_loss, test_time = test_model(model, test_loader, device, Loss)
        print('test loss :', test_loss, 'time :', test_time)



def contrast_train(model, train_loader, optimizer, classification_criterion):
    model.train()
    for i, (text, audio, video, labels) in enumerate(train_loader):
        
        output, total_loss = model(text, audio, video)
        
        # calculate the classification loss
        classification_loss = classification_criterion(output, labels)
        
        # calculate the total loss
        loss = classification_loss + total_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Step [{}/{}], Loss: {:.4f}' 
                   .format(i+1, len(train_loader), loss.item()))

def contrast_test(model, test_loader, classification_criterion):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for text, audio, video, labels in test_loader:
            output, total_loss = model(text, audio, video)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            classification_loss = classification_criterion(output, labels)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    
    
    
        
    