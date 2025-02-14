import matplotlib.pyplot as plt
import pandas as pd



def accuracy_loss_graphs(log_csv):

    df = pd.read_csv(log_csv)

    plt.style.use('ggplot')

    plt.subplot(1,2,1)
    plt.plot(df['accuracy'])
    plt.plot(df['val_accuracy'])
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['train accuracy', 'validation accuracy'], loc='best', prop={'size': 12})

    plt.subplot(1,2,2)
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'validation loss'], loc='best', prop={'size': 12})

    plt.show()



if __name__ == "__main__":

    log_csv = r'saved-models/bcd-final/bcd-final-fit-log.csv'

    accuracy_loss_graphs(log_csv)
