#!/usr/bin/env python

# author: aa4461

import sys

import numpy as np
import scoring_utils as su
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix as get_confusion_matrix


def main():
    utt2labels_filepath = sys.argv[1]
    predictions_ark_filepath = sys.argv[2]
    output_path = sys.argv[3]

    utt2labels = su.parse_utt2labels(utt2labels_filepath)
    utt2predictions = su.parse_predictions_ark(predictions_ark_filepath)

    y_true = []
    y_pred = []
    for utt in (utt2labels.keys() | utt2predictions.keys()):
        y_true.append(utt2labels[utt])
        y_pred.append(np.argmax(utt2predictions[utt]))

    accuracy = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    confusion_matrix = get_confusion_matrix(y_true, y_pred)

    with open(output_path, 'w') as f:
        f.write("Accuracy: %s\n" % (accuracy))
        f.write("Micro-F1: %s\n" % (micro_f1))
        f.write("Macro-F1: %s\n" % (macro_f1))

        f.write("Confusion matrix:\n")
        for row in confusion_matrix:
            f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
