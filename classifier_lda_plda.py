import pickle
from ast import literal_eval
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

from plda.classifier import Classifier

__author__ = "Kailash Karthik S, Jessica Huynh"
__uni__ = "ks3740, jyh2127"
__email__ = "kailashkarthik.s@columbia.edu, jyh2127@columbia.edu"
__status__ = "Development"


class EmbeddingSource(Enum):
    TEXT = 'TextEmbeddings'
    SPEECH = 'SpeechEmbeddings'
    MULTIMODAL = 'MultimodalEmbeddings'


class LdaPLdaClassifier:
    def __init__(self, data_file, embedding_source: EmbeddingSource, n_components=200, load_pre_trained_model=False):
        input_, output_ = self.load_data(data_file, embedding_source)
        self.embedding_source = embedding_source
        self.load_pre_trained_model = load_pre_trained_model
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_data_splits(input_, output_)
        self.x_train_reduced = None
        self.x_test_reduced = None
        if load_pre_trained_model:
            self.lda = pickle.load(
                open('empirical_results/models/meld_lda_' + str(self.embedding_source.value) + '.pkl', 'rb'))
            self.p_lda = pickle.load(
                open('empirical_results/models/meld_p_lda_' + str(self.embedding_source.value) + '.pkl', 'rb'))
        else:
            self.lda = LinearDiscriminantAnalysis(n_components=n_components)
            self.p_lda = Classifier()

    @staticmethod
    def load_data(data_file, embedding_source: EmbeddingSource):
        df = pd.read_csv(data_file)
        input_ = df[embedding_source.value].apply(literal_eval)
        output_ = df['Emotion_Label']
        return input_, output_

    @staticmethod
    def get_data_splits(input_, output_):
        mask = np.random.rand(len(input_)) < 0.8
        x_train = input_[mask]
        y_train = output_[mask]
        x_test = input_[~mask]
        y_test = output_[~mask]
        shuffled_train = shuffle(x_train, y_train)
        x_train, y_train = shuffled_train
        shuffled_test = shuffle(x_test, y_test)
        x_test, y_test = shuffled_test
        return x_train.to_list(), y_train.to_list(), x_test.to_list(), y_test.to_list()

    def train_lda(self, save_model=False):
        self.lda.fit(self.x_train, self.y_train)
        if save_model:
            with open('empirical_results/models/meld_lda_' + str(self.embedding_source.value) + '.pkl', 'wb') as file_:
                pickle.dump(self.lda, file_)

    def reduce_input_dimensions_using_lda(self):
        self.train_lda()
        self.x_train_reduced = self.lda.transform(self.x_train)
        self.x_test_reduced = self.lda.transform(self.x_test)

    def train_p_lda(self, save_model=False):
        self.p_lda.fit_model(self.x_train_reduced, self.y_train)
        if save_model:
            with open('empirical_results/models/meld_p_lda_' + str(self.embedding_source.value) + '.pkl',
                      'wb') as file_:
                pickle.dump(self.p_lda, file_)

    def get_predictions(self, input_):
        return self.p_lda.predict(input_, normalize_logps=True)

    def evaluate_model_for_input(self, input_, output_, input_title):
        y_hat, y_hat_log_prob = self.get_predictions(input_)
        accuracy = accuracy_score(output_, y_hat)
        f1 = f1_score(output_, y_hat, average='macro')
        print('Performance - ' + input_title)
        print('Accuracy : ' + str(accuracy))
        print('F1 Score : ' + str(f1))

    def train_and_evaluate_model(self):
        if not self.load_pre_trained_model:
            self.train_lda(True)
            self.reduce_input_dimensions_using_lda()
            self.train_p_lda(True)
        self.evaluate_model()

    def evaluate_model(self):
        self.evaluate_model_for_input(self.x_train_reduced, self.y_train, 'Train')
        self.evaluate_model_for_input(self.x_test_reduced, self.y_test, 'Test')


def train_on_meld():
    data_file = 'meld/data/dataset_with_multi_modal_embeddings.csv'
    classifier = LdaPLdaClassifier(data_file, EmbeddingSource.TEXT)
    classifier.train_and_evaluate_model()


def evaluate_on_iemocap():
    data_file = 'iemocap/data/dataset_with_multi_modal_embeddings.csv'
    classifier = LdaPLdaClassifier(data_file, EmbeddingSource.MULTIMODAL, load_pre_trained_model=True)
    classifier.reduce_input_dimensions_using_lda()
    classifier.evaluate_model()


if __name__ == '__main__':
    train_on_meld()
    evaluate_on_iemocap()
