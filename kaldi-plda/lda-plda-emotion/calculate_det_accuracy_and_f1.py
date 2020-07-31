import csv
import pickle
from collections import defaultdict
from optparse import OptionParser

import numpy as np
from pyannote.metrics.plot.binary_classification import plot_det_curve
from sklearn.metrics import classification_report, confusion_matrix


def get_label(utterance):
    return utterance.split('-')[0]


def get_emotion_with_max_score(scores_by_emotion):
    max_score_by_emotion = {emotion: max(scores) \
                            for emotion, scores in scores_by_emotion.items()}
    return list(sorted(max_score_by_emotion.items(), key=lambda emotion_and_score: emotion_and_score[1]))[-1]


def get_emotion_with_max_avg_score(scores_by_emotion):
    avg_score_by_emotion = {emotion: np.mean(scores) \
                            for emotion, scores in scores_by_emotion.items()}
    return list(sorted(avg_score_by_emotion.items(), key=lambda emotion_and_score: emotion_and_score[1]))[-1]


def main():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        '-v',
        '--variant',
        dest='variant'
    )
    parser.add_option(
        '-t',
        '--trials-file',
        dest='trials_file'
    )
    parser.add_option(
        '-s',
        '--score-file',
        dest='score_file'
    )
    parser.add_option(
        '-o',
        '--output-dir',
        dest='output_dir'
    )
    (options, args) = parser.parse_args()

    labels_and_scores_by_trial = defaultdict(dict)
    with open(options.trials_file, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            train_utterance, test_utterance, label = row
            labels_and_scores_by_trial[(train_utterance, test_utterance)]['label'] = label

    test_scores = defaultdict(lambda: defaultdict(list))
    with open(options.score_file, 'r') as f:
        for row in csv.reader(f, delimiter=' '):
            train_utterance, test_utterance, score = row
            labels_and_scores_by_trial[(train_utterance, test_utterance)]['score'] = float(score)
            test_scores[test_utterance][get_label(train_utterance)].append(float(score))

    det_labels = []
    det_scores = []
    for key in labels_and_scores_by_trial.keys():
        trial = labels_and_scores_by_trial[key]
        if 'label' not in trial or 'score' not in trial:
            continue
        det_labels.append(True if trial['label'] == 'target' else False)
        det_scores.append(float(trial['score']))

    eer = plot_det_curve(det_labels, det_scores, '%s/%s' % (options.output_dir, options.variant))
    print("EER=%s" % (eer))

    real_labels = []
    max_score_labels = []
    max_avg_score_labels = []
    for test_utterance, scores_by_emotion in test_scores.items():
        if 'test' not in test_utterance:
            print('Are you sure %s is a test utterance?' % (test_utterance))

        real_labels.append(get_label(test_utterance))
        max_score_labels.append(get_emotion_with_max_score(scores_by_emotion)[0])
        max_avg_score_labels.append(get_emotion_with_max_avg_score(scores_by_emotion)[0])

    print('Classification using max score:')
    max_score_classification_report = classification_report(real_labels, max_score_labels, output_dict=True)
    with open('%s/max_score_classification_report.pkl' % options.output_dir, 'wb') as f:
        pickle.dump(max_score_classification_report, f)
    max_score_confusion_matrix = confusion_matrix(real_labels, max_score_labels)
    with open('%s/max_score_confusion_matrix.pkl' % options.output_dir, 'wb') as f:
        pickle.dump(max_score_confusion_matrix, f)
    print(classification_report(real_labels, max_score_labels))

    print('Classification using max avg score:')
    avg_score_classification_report = classification_report(real_labels, max_avg_score_labels, output_dict=True)
    with open('%s/avg_score_classification_report.pkl' % options.output_dir, 'wb') as f:
        pickle.dump(avg_score_classification_report, f)
    avg_score_confusion_matrix = confusion_matrix(real_labels, max_avg_score_labels)
    with open('%s/avg_score_confusion_matrix.pkl' % options.output_dir, 'wb') as f:
        pickle.dump(avg_score_confusion_matrix, f)
    print(classification_report(real_labels, max_avg_score_labels))


if __name__ == "__main__":
    main()
