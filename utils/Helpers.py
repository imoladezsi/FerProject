import numpy as np
import matplotlib.pyplot as plt
import os

class Helpers(object):
    pass

    @staticmethod
    def save_figure(history, epochs, whereto, save_as):
        print("[INFO] saving plot...")
        N = np.arange(0, epochs)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, history.history["loss"], label="train_loss")
        plt.plot(N, history.history["val_loss"], label="val_loss")
        plt.plot(N, history.history["acc"], label="train_acc")
        plt.plot(N, history.history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy " + save_as)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(os.path.join(whereto, save_as + "_plot.png"))

