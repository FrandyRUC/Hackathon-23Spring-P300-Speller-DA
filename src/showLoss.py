import pandas as pd
import matplotlib.pyplot as plt

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
    plt.title('Loss and Validation Loss vs. Epoch')
    plt.legend()
    plt.show()
printLossTable()
#printLossTable('testLos.csv')
