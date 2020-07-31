# author: aa4461

from functools import reduce

import numpy as np
import pandas as pd
from kaldiio import ReadHelper


class ScoringResults:
    def __init__(self, name):
        self.name = name
        self.missing_label = []
        self.missing_prediction = []
        self.confusion_matrix = np.zeros((5, 5))

    def add_missing_label(self, missing_label):
        self.missing_label.append(missing_label)

    def add_missing_prediction(self, missing_prediction):
        self.missing_prediction.append(missing_prediction)

    def get_accuracy(self):
        correct = 0.0
        incorrect = 0.0
        for r in range(5):
            for c in range(5):
                if r == c:
                    correct += self.confusion_matrix[r][c]
                else:
                    incorrect += self.confusion_matrix[r][c]
        return correct / (correct + incorrect)

    def print_results(self):
        print(self.name)
        print('Overall accuracy: %s' % (self.get_accuracy()))

        row_labels = ['A: anger/disgust', 'A: fear/surprise', 'A: happiness', 'A: neutral', 'A: sadness']
        col_labels = ['P: anger/disgust', 'P: fear/surprise', 'P: happiness', 'P: neutral', 'P: sadness']
        print(pd.DataFrame(self.confusion_matrix, index=row_labels, columns=col_labels))

        print('Missing label: %s' % (self.missing_label))
        print('Missing prediction: %s' % (self.missing_prediction))

    def save_results(self, output_path):
        np.savetxt(('%s/%s.out') % (output_path, self.name), self.confusion_matrix)


def parse_utt2labels(filepath):
    utt2labels = {}
    with open(filepath) as f:
        for line in f:
            utterance_id, label = line.split(' ')
            utt2labels[utterance_id] = int(label)
    return utt2labels


def parse_predictions_ark(filepath):
    utt2predictions = {}
    with ReadHelper('ark:%s' % (filepath)) as reader:
        for utterance_id, predictions in reader:
            if utterance_id in utt2predictions:
                raise Exception('%s duped in %s' % (utterance_id, filepath))
            utt2predictions[utterance_id] = predictions
    return utt2predictions


def get_most_frequent_emotion(predictions_by_frame):
    emotion_counts = np.zeros(5)
    for prediction in predictions_by_frame:
        most_likely_emotion = np.argmax(prediction)
        emotion_counts[most_likely_emotion] += 1
    return np.argmax(emotion_counts)


def get_most_likely_emotion(predictions_by_frame):
    emotion_log_likelihoods = np.sum(predictions_by_frame, axis=0)
    return np.argmax(emotion_log_likelihoods)


def get_most_likely_weighted_emotion(weights, predictions_by_frame):
    emotion_log_likelihoods = np.sum(predictions_by_frame, axis=0)
    weighted = np.multiply(emotion_log_likelihoods, weights)
    return np.argmax(weighted)


def get_longest_streak_emotion(predictions_by_frame):
    emotion_streaks = np.zeros(5)
    current_streak = 1
    current_emotion = np.argmax(predictions_by_frame[0])
    for prediction in predictions_by_frame[1:]:
        emotion_streaks[current_emotion] = max(current_streak, emotion_streaks[current_emotion])

        most_likely_emotion = np.argmax(prediction)
        if current_emotion != most_likely_emotion:
            current_streak = 1
            current_emotion = most_likely_emotion
        else:
            current_streak += 1
    emotion_streaks[current_emotion] = max(current_streak, emotion_streaks[current_emotion])
    return np.argmax(emotion_streaks)


def score(experiment_name, utt2labels, utt2predictions, resolution_method):
    scoring_results = ScoringResults(experiment_name)
    for utterance in reduce(set.union, (set(d.keys()) for d in [utt2labels, utt2predictions])):
        label = utt2labels[utterance]
        predictions_by_frame = utt2predictions[utterance] if utterance in utt2predictions else []

        if label == None:
            scoring_results.add_missing_label(utterance)
        elif len(predictions_by_frame) == 0:
            scoring_results.add_missing_prediction(utterance)
        else:
            prediction = resolution_method(predictions_by_frame)
            scoring_results.confusion_matrix[label, prediction] += 1
    return scoring_results
