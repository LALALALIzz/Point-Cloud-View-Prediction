import numpy as np
import matplotlib.pyplot as plt

class ResultVisualization:

    # modelType:{GRU_Dense, GRU_En_De, Transformer}
    # predictionLen: 1-5s
    # mode: Angular or Position
    def __init__(self, mode, architecture, pred_step):
        self.mode = mode
        self.architecture = architecture
        self.pred_step = pred_step

    def loss_plot(self, train_loss, test_loss):
        plt.figure()
        plt.title("%s Train Test Loss for %s in period of %d points" % (self.architecture, self.mode, self.pred_step))
        epoch_num = len(train_loss)
        epochs = np.arange(1, epoch_num+1)
        plt.plot(epochs, train_loss, 'b', label='Train loss')
        plt.plot(epochs, test_loss, 'r', label='Test loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # zoomInRange is 0 in default, the plot will show the whole range
    # If zoomInRange is not 0, plot will show the given range
    # zoomInBias is 0 in default, the plot will begin from index 0
    # If zoomInBias is given and not equal to 0, the plot will start from the index %zoomInBias

    def data_plot(self, groundtruth, prediction, zoomInRange, zoomInBias):
        plt.figure()
        if zoomInRange != 0:
            samples_index = np.arange(zoomInBias, zoomInBias + zoomInRange)
        else:
            samples_index = np.arange(0, len(groundtruth))
        if self.mode == "angle":
            subtitles = ["pitch", "yaw", "roll"]
        elif self.mode == "position":
            subtitles = ["x-axis", "y-axis", "z-axis"]
        Ylabel = 'degree' if self.mode == "Angular" else 'unit'
        plt.title("%s prediction for %s in period of %ds" % (self.architecture, self.mode, self.pred_step))
        plt.subplot(3, 1, 1)
        plt.title(subtitles[0])
        plt.plot(samples_index, groundtruth[:, 0], 'b', label='GroundTruth')
        plt.plot(samples_index, prediction[:, 0], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 2)
        plt.title(subtitles[1])
        plt.plot(samples_index, groundtruth[:, 1], 'b', label='GroundTruth')
        plt.plot(samples_index, prediction[:, 1], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 3)
        plt.title(subtitles[2])
        plt.plot(samples_index, groundtruth[:, 2], 'b', label='GroundTruth')
        plt.plot(samples_index, prediction[:, 2], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.legend()
        plt.show()