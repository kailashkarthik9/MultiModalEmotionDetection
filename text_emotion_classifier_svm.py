from ast import literal_eval

import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


class TextEmotionClassifier:
    def __init__(self, test_files, train_files, dev_files):
        self.test_data = self.load_data(test_files)
        self.train_data = self.load_data(train_files)
        self.dev_data = self.load_data(dev_files)

    @staticmethod
    def load_data(data_files):
        df = None
        for data_file in data_files:
            if df is None:
                df = pd.read_csv(data_file)
            else:
                df.append(pd.read_csv(data_file))
        input_ = df['TextEmbeddings'].apply(literal_eval)
        output_ = df['Emotion_Label']
        return input_, output_

    def tune_model(self):
        """
        This is a utility method that is used to tune the model by running grid search. A developer utility!
        :param x: The input to the model
        :param y: The true labels expected from the model
        :return: None
        """
        model_grid = svm.SVC()
        # param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['linear', 'rbf', 'poly'],
        #               'degree': [0, 1, 2, 3, 4, 5, 6], 'gamma': [0.1, 1, 10, 100]}
        param_grid = {'gamma': [0.1, 1, 10, 100]}
        grid = GridSearchCV(estimator=model_grid, param_grid=param_grid)
        grid_result = grid.fit(self.train_data[0].tolist(), self.train_data[1].tolist())
        print('Best Score: ', grid_result.best_score_)
        print('Best Params: ', grid_result.best_params_)

    def train_test_model(self, alpha, fit_prior, kernel, C, degree, gamma):
        model_svm = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        model_svm.fit(self.train_data[0].tolist(), self.train_data[1].tolist())
        y_pred_svm = model_svm.predict(self.dev_data[0].tolist())
        accuracy_svm = accuracy_score(self.dev_data[1].tolist(), y_pred_svm)
        f1_svm = f1_score(self.dev_data[1].tolist(), y_pred_svm, average='macro')
        print('SVM')
        print('Accuracy : ' + str(accuracy_svm))
        print('F1 Score : ' + str(f1_svm))


if __name__ == '__main__':
    meld_train = 'meld/data/train_sent_emo_norm_word_embedded.csv'
    meld_test = 'meld/data/test_sent_emo_norm_word_embedded.csv'
    meld_dev = 'meld/data/dev_sent_emo_norm_word_embedded.csv'
    classifier = TextEmotionClassifier([meld_test], [meld_train], [meld_dev])
    # classifier.tune_model()
    classifier.train_test_model(alpha=0.1, fit_prior=True, C=10, kernel='rbf', degree=0, gamma=0.1)
