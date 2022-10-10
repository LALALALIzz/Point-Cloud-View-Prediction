import numpy as np
import matplotlib.pyplot as plt

class ResultVisualization:
    """
    modelType:{GRU_Dense, GRU_En_De, Transformer}
    observ_step is the length of users' past history trajectory we used.
    pred_step is the length of prediction
    mode: Angular or Position
    """
    def __init__(self, mode, architecture, observ_step, pred_step):
        self.mode = mode
        self.architecture = architecture
        self.observ_step = observ_step
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

    """
    zoomInRange is 0 in default, the plot will show the whole range
    If zoomInRange is not 0, plot will show the given range
    zoomInBias is 0 in default, the plot will begin from index 0
    If zoomInBias is given and not equal to 0, the plot will start from the index %zoomInBias
    We limit the range of y-aixs of the graph according to the min and max of the whole sequence.
    ymin and ymax should be an array in size of 3
    """
    def data_plot(self, groundtruth, prediction, zoomInRange, zoomInBias, mean, std):
        plt.figure()
        if zoomInRange != 0:
            groundtruth_index = np.arange(zoomInBias, self.observ_step + zoomInBias + zoomInRange)
            pred_index = np.arange(self.observ_step + zoomInBias, self.observ_step + zoomInBias + zoomInRange)
        else:
            groundtruth_index = np.arange(0, self.observ_step + len(prediction))
            pred_index = np.arange(self.observ_step, self.observ_step + len(prediction))
        if self.mode == "angle":
            subtitles = ["pitch", "yaw", "roll"]
        elif self.mode == "position":
            subtitles = ["x-axis", "y-axis", "z-axis"]

        # Data denormalization

        groundtruth = groundtruth * std + mean
        prediction = prediction * std + mean

        Ylabel = 'degree' if self.mode == "Angular" else 'unit'
        plt.title("%s prediction for %s in period of %ds" % (self.architecture, self.mode, self.pred_step))
        plt.subplot(3, 1, 1)
        # plt.ylim(ymin[0], ymax[0])
        plt.title(subtitles[0])
        plt.plot(groundtruth_index, groundtruth[:, 0], 'b', label='GroundTruth')
        plt.plot(pred_index, prediction[:, 0], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 2)
        # plt.ylim(ymin[1], ymax[1])
        plt.title(subtitles[1])
        plt.plot(groundtruth_index, groundtruth[:, 1], 'b', label='GroundTruth')
        plt.plot(pred_index, prediction[:, 1], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.subplot(3, 1, 3)
        # plt.ylim(ymin[2], ymax[2])
        plt.title(subtitles[2])
        plt.plot(groundtruth_index, groundtruth[:, 2], 'b', label='GroundTruth')
        plt.plot(pred_index, prediction[:, 2], 'r', label='Prediction')
        plt.xlabel('sample index')
        plt.ylabel(Ylabel)
        plt.legend()
        plt.show()