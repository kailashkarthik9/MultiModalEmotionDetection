import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


class EmoVoxCelebDatasetFormatter:
    """
    This class is used to format the EmoVoxCeleb data
    """

    def __init__(self):
        self.logits_utterance_major = self.load_logits_data()  # Frame major actually
        self.logits_emotion_major = self.reorder_logits_dataset()

    @staticmethod
    def load_logits_data():
        """
        This method loads the logits data from the teacher emovoxceleb model
        :return: A list of lists containing the logits
        """
        x = loadmat('data/senet50-ferplus-logits.mat')
        return x['wavLogits'][0]

    def reorder_logits_dataset(self):
        """
        This method reorders the dataset to transform it from utterance-major to emotion-major format
        :return: The logits in a emotion-major format
        """
        logits_matrix = np.empty((len(self.logits_utterance_major), 8), dtype=object)
        for utterance_idx, utterance in enumerate(logits_matrix):
            for emotion_idx, emotion in enumerate(utterance):
                logits_matrix[utterance_idx][emotion_idx] = []

        for utterance_idx, utterance in enumerate(self.logits_utterance_major):
            for frame_idx, frame in enumerate(utterance):
                for emotion_idx, emotion in enumerate(frame):
                    logits_matrix[utterance_idx][emotion_idx].append(emotion)
        return logits_matrix

    def get_emotion_statistics(self):
        """
        This method gets the emotion-level statistics for each utterance in the dataset
        :return: The statistics matrix
        """
        statistics_matrix = np.empty((len(self.logits_emotion_major), 8, 3))
        for utterance_idx, utterance in enumerate(self.logits_emotion_major):
            for emotion_idx, emotion in enumerate(utterance):
                statistics_matrix[utterance_idx][emotion_idx][0] = np.mean(
                    self.logits_emotion_major[utterance_idx][emotion_idx])
                statistics_matrix[utterance_idx][emotion_idx][1] = np.std(
                    self.logits_emotion_major[utterance_idx][emotion_idx])
                statistics_matrix[utterance_idx][emotion_idx][2] = max(
                    self.logits_emotion_major[utterance_idx][emotion_idx])
        return statistics_matrix

    def get_utterance_statistics(self):
        """
        This method gets the utterance-level statistics in the dataset
        :return: The statistics matrix
        """
        statistics_matrix = np.empty((len(self.logits_emotion_major), 3))
        for utterance_idx, utterance in enumerate(self.logits_emotion_major):
            print('Utterance : ' + str(utterance_idx))
            utterance_means = [emotion[0] for emotion_idx, emotion in enumerate(self.emotion_statistics[utterance_idx])]
            utterance_maxs = sorted(
                [emotion[2] for emotion_idx, emotion in enumerate(self.emotion_statistics[utterance_idx])])
            statistics_matrix[utterance_idx][0] = np.std(utterance_means)
            statistics_matrix[utterance_idx][1] = np.std(utterance_maxs)
            statistics_matrix[utterance_idx][2] = utterance_maxs[-1] - utterance_maxs[-2]
        return statistics_matrix

    def create_emotion_scatter_plot(self, emotion_index):
        """
        This method generates the emotion-level visualizations
        :param emotion_index: The emotion for which visualization is to be generated
        :return: None
        """
        means = [emotion[0]
                 for utterance_idx, utterance in enumerate(self.emotion_statistics)
                 for emotion_idx, emotion in enumerate(utterance)
                 if emotion_idx == emotion_index]
        std_devs = [emotion[1]
                    for utterance_idx, utterance in enumerate(self.emotion_statistics)
                    for emotion_idx, emotion in enumerate(utterance)
                    if emotion_idx == emotion_index]
        max_values = [emotion[2]
                      for utterance_idx, utterance in enumerate(self.emotion_statistics)
                      for emotion_idx, emotion in enumerate(utterance)
                      if emotion_idx == emotion_index]
        x = [x for x in range(len(means))]
        # plt.scatter(x, max_values, color='green')
        # plt.scatter(x, means, color='red')
        plt.scatter(x, std_devs, color='blue')
        plt.xlabel('Utterance')
        plt.ylabel('Emotion Statistics')
        # plt.show()
        plt.savefig('emotion' + str(emotion_index) + '.png')

    def create_utterance_scatter_plot(self):
        """
        This method generates utterance-level visualizations from the dataset
        :return: None
        """
        means = [utterance[0] for utterance in self.utterance_statistics]
        maxs = [utterance[1] for utterance in self.utterance_statistics]
        max_diffs = [utterance[2] for utterance in self.utterance_statistics]
        x = [x for x in range(len(means))]
        # plt.scatter(x, means, color='red')
        # plt.xlabel('Utterance')
        # plt.ylabel('Utterance Statistics')
        # plt.show()
        # plt.savefig('means.png')
        # plt.scatter(x, maxs, color='blue')
        # plt.xlabel('Utterances')
        # plt.ylabel('Mean of the highest scoring emotion across the utterance\'s length')
        # plt.show()
        plt.savefig('maxs.png')
        plt.scatter(x, max_diffs, color='blue')
        plt.xlabel('Utterances')
        plt.ylabel('Difference between top 2 highest scoring emotions')
        # plt.show()
        plt.savefig('max_diffs.png')

    def create_max_emotion_scatter_plot(self):
        """
        This method generates the visualization for the highest scoring emotion for each utterance
        :return: None
        """
        std_devs = []
        for utterance_idx, utterance in enumerate(self.emotion_statistics):
            max = float('-inf')
            std_dev = 0
            for emotion_idx, emotion in enumerate(utterance):
                if emotion[0] > max:
                    max = emotion[2]
                    std_dev = emotion[1]
            std_devs.append(std_dev)
        x = [x for x in range(len(std_devs))]
        plt.scatter(x, std_devs, color='blue')
        plt.xlabel('Utterances')
        plt.ylabel('Standard Deviations Distribution for Emotion with Highest Mean')
        plt.savefig('best_emotion_using_max.png')


if __name__ == '__main__':
    formatter = EmoVoxCelebDatasetFormatter()
    # formatter = pickle.load(open('voxceleb.obj', 'rb'))
    formatter.emotion_statistics = formatter.get_emotion_statistics()
    formatter.utterance_statistics = formatter.get_utterance_statistics()
    # for index in range(8):
    #     formatter.create_emotion_scatter_plot(index)
    formatter.create_utterance_scatter_plot()
    formatter.create_max_emotion_scatter_plot()
    # pickle.dump(formatter, open('voxceleb.obj', 'wb'))
