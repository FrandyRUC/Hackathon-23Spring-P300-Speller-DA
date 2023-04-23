'''
   print loss and accuracy.
   data is from the file 'logs.csv'
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def printLossTable(fileName="logs.csv"):
    # Read the CSV file
    df = pd.read_csv(fileName)
    #size_len=30
    #stopP=size_len-1

    # Extract "epoch" and "val_loss" columns
    epoch = df['epoch']
    val_loss = df['val_loss']
    loss=df['loss']
    # Plot the graph using Seaborn
    sns.lineplot(x='epoch', y='val_loss', data=df, label='Validation Loss')
    sns.lineplot(x='epoch', y='loss', data=df, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss(CNN)')
    plt.legend()
    plt.show()
def printAccuracy(fileName="logs.csv"):
    # Read the CSV file
    df = pd.read_csv(fileName)
    #size_len=30
    #stopP=size_len-1

    # Extract "epoch" and "val_acc" columns
    epoch = df['epoch']
    val_acc = df['val_accuracy']
    accuracy=df['accuracy']
    # Plot the graph using Seaborn
    sns.lineplot(x='epoch', y='val_accuracy', data=df, label='val_acc')
    sns.lineplot(x='epoch', y='accuracy', data=df, label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy(CNN)')
    plt.legend()
    plt.show()
printLossTable()
printAccuracy()
#printAccuracy()
#printLossTable('testLos.csv')
