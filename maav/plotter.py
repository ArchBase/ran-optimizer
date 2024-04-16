import pickle
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import matplotlib.pyplot as plt

def plot():
    
    loss = None
    val_loss = None

    with open("saved_model/training_log/loss.history", 'rb') as file:
        loss = pickle.load(file)

    with open("saved_model/training_log/val_loss.history", 'rb') as file:
        try:
            val_loss = pickle.load(file)
        except EOFError:
            pass


    epochs = range(1, len(loss) + 1)


    # Plot loss vs. epoch
    plt.plot(epochs, loss, label='Training Loss')
    if val_loss != None:
        plt.plot(epochs, val_loss, label='Validation Loss')

    # Label the plot
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()
    #plt.savefig('plot.png')
