import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def model_par_to_device(batch, device):
    '''
    input: torch batch
    output: torch arrays
    function: divides all columns into single ones and makes sure to use the right device
    '''
    sender_input_ids = batch['sender_input_ids'].to(device)
    sender_attention_mask = batch['sender_attention_mask'].to(device)
    subject_input_ids = batch['subject_input_ids'].to(device)
    subject_attention_mask = batch['subject_attention_mask'].to(device)
    text_input_ids = batch['text_input_ids'].to(device)
    text_attention_mask = batch['text_attention_mask'].to(device)
    labels = batch['label'].to(device)
    return sender_input_ids,sender_attention_mask,subject_input_ids,subject_attention_mask,text_input_ids,text_attention_mask,labels

def model_training_loop(model, train_loader, optimizer, loss_function, device):
    '''
    input: model (nn.Model), training loader (Torch datasetloader), optimizer (predefined!, Torch), loss function (predefined!, Torch)
    output: training loss (scalar)
    function: loops over the batches in the training loader and trains it by optimizing it using the loss function
    '''
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader):       
        sender_input_ids,sender_attention_mask,subject_input_ids,subject_attention_mask,text_input_ids,text_attention_mask,labels = model_par_to_device(batch,device)
        optimizer.zero_grad()       
        outputs = model(sender_input_ids, sender_attention_mask, subject_input_ids, subject_attention_mask, text_input_ids, text_attention_mask)       
        loss_train = loss_function(outputs, labels)
        loss_train.backward()
        optimizer.step()     
        total_train_loss += loss_train.item()
    return total_train_loss

def model_testing_loop(model, test_loader, device):
    '''
    input: model (nn.Model), testing loader (Torch datasetloader)
    output: accuracy (scalar in percent)
    function: loops over the batches in the testing loader and evaluates the prediction and the accuracy
    '''
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:           
            sender_input_ids,sender_attention_mask,subject_input_ids,subject_attention_mask,text_input_ids,text_attention_mask,labels = model_par_to_device(batch,device)         
            outputs = model(sender_input_ids, sender_attention_mask, subject_input_ids, subject_attention_mask, text_input_ids, text_attention_mask)
            predicted = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted)    
    return accuracy_score(y_true, y_pred)
