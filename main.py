import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from email_text_editor import *
from transformer_model import *
from Data_Cleaner import *
from learning_loop_simp import *

'''
    For your notice, make sure, that you have .csv file containg 4 columns:
    sender, subject, textcontent, label
    in which each row stands for a mail.
    if this is not the case, you should adjust the code properly.
'''


if __name__ == '__main__':

    # Load dataset: columns 'sender', 'subject', 'text' as csv
    dir_path = os.path.dirname(__file__)
    file_name = 'your filename'
    file_path = os.path.join(dir_path,file_name)
    data = pd.read_csv(file_path,delimiter='|',encoding = "ISO-8859-1") # Please check the delimiter and encoding, can be adjusted regarding the data

    # Cleaning Data
    data.columns = ["sender", "subject", "text","label"]
    data['sender'] = data['sender'].transform(lambda x:take_only_adresses(x))
    data = data.fillna('')
    data['text'] = data['text'].transform(lambda x:get_rid_of_symbols(x))
    data['subject'] = data['subject'].transform(lambda x:get_rid_of_all_symbols(x))
    data['text'] = data['text'].transform(lambda x:delete_not_decoded_text(x))
    data['label'].values[:] = 0 # set all spam mail labels to not spam = 0

    spam_labeler(data) # Label the spam mails

    # to observe the sender more clearly, all symbos are deleted
    data['sender'] = data['sender'].transform(lambda x:get_rid_of_all_symbols(x))

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Datasets and dataloaders + tokenizer with BERT
    max_len = 128
    train_dataset = EMAIL_Dataset(train_data, max_len)
    test_dataset = EMAIL_Dataset(test_data, max_len)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model parameters
    input_dim = 3 * 768  # Concatenate sender, subject, and text BERT embeddings
    hidden_dim = 256
    num_classes = 2

    # Initialize model, optimizer and loss function
    model = BERT_Transformer(input_dim, hidden_dim, num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    loss_function = nn.CrossEntropyLoss()

    # Set to right device (important for information exchange)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set lists for plotting
    loss_train_list = []
    accuracy_test_list = []

    # Set # of epochs
    num_epochs = 3

    for epoch in range(num_epochs):
        # Train the model
        total_train_loss = model_training_loop(model, train_loader, optimizer, loss_function, device)
        
        loss_train_list.append(total_train_loss)
        print(f'Train Epoch {epoch+1}/{num_epochs}, Loss: {total_train_loss/len(train_loader)}')
    
        # Evaluate on test set on trained epoch
        accuracy = model_testing_loop(model, test_loader, device)
        accuracy_test_list.append(accuracy)
        print(f'Test Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy}')
