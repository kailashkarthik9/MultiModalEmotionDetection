#!/usr/bin/env python

# author: aa4461

import random
import sys
from collections import defaultdict
from functools import reduce

import numpy as np
import scoring_utils as su
from sklearn.linear_model import LogisticRegression


def get_sessions_to_utts(utt2labels, utt2predictions):
    sessions_to_utts = defaultdict(list)
    for utt in reduce(set.union, (set(d.keys()) for d in [utt2labels, utt2predictions])):
        source_session = None
        for session in range(1, 6):
            session_id = 'Ses0%s' % (session)
            if session_id in utt:
                source_session = session
                break

        if source_session == None:
            raise Exception('%s missing session id' % (utt))
        else:
            sessions_to_utts[source_session].append(utt)
    return sessions_to_utts


def subset_predictions_and_emotions_equally(utts, utt2labels, utt2predictions):
    emotion_counts = np.zeros(5)
    for utt in utts:
        emotion = utt2labels[utt]
        emotion_counts[emotion] += 1
    min_emotion_count = np.min(emotion_counts)

    random.shuffle(utts)

    predictions = []
    emotions = []
    emotion_counts = np.zeros(5)
    for utt in utts:
        emotion = utt2labels[utt]
        if emotion_counts[emotion] < min_emotion_count:
            predictions.append(utt2predictions[utt])
            emotions.append(utt2labels[utt])
            emotion_counts[emotion] += 1

    return (np.array(predictions), np.array(emotions))


def get_most_likely_weighted_emotion(classifier):
    def reduction_function(predictions_by_frame):
        return classifier.predict([predictions_by_frame])

    return reduction_function


def main():
    utt2labels_filepath = sys.argv[1]
    predictions_ark_filepath = sys.argv[2]

    utt2labels = su.parse_utt2labels(utt2labels_filepath)
    utt2predictions = {utt: np.sum(predictions, axis=0) for utt, predictions in
                       su.parse_predictions_ark(predictions_ark_filepath).items()}

    sessions_to_utts = get_sessions_to_utts(utt2labels, utt2predictions)
    for session in range(1, 6):
        predictions_x, emotions_y = subset_predictions_and_emotions_equally(
            sessions_to_utts[session], utt2labels, utt2predictions
        )
        # other classifiers that you can try
        # classifier = LinearDiscriminantAnalysis()
        # classifier = svm.LinearSVC()
        classifier = LogisticRegression(random_state=0, multi_class='auto', solver='lbfgs', max_iter=1000)
        classifier.fit(predictions_x, emotions_y)

        scores = su.score(
            'Session %s' % (session),
            {utt: label for (utt, label) in utt2labels.items() if ('Ses0%s' % (session)) not in utt},
            {utt: prediction for (utt, prediction) in utt2predictions.items() if ('Ses0%s' % (session)) not in utt},
            get_most_likely_weighted_emotion(classifier)
        )
        scores.print_results()


if __name__ == "__main__":
    main()
