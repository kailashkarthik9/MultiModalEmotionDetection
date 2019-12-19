from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics import f1_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

__author__ = "Kailash Karthik S"
__uni__ = "ks3740"
__email__ = "kailashkarthik.s@columbia.edu"
__status__ = "Development"


USE_CUDA = torch.cuda.is_available()


class EmotionClassifierNetwork(nn.Module):

    def __init__(self):
        super(EmotionClassifierNetwork, self).__init__()
        self.fc_layer_1 = nn.Linear(768, 150)
        self.fc_layer_2 = nn.Linear(128, 5)

    def forward(self, sentence_embedding):
        z_1 = self.fc_layer_1(sentence_embedding)
        a_1 = func.relu(z_1)
        z_2 = self.fc_layer_2(a_1)
        a_2 = func.softmax(z_2)
        return a_2


class TextEmotionClassifier:
    def __init__(self, test_files, train_files, dev_files, batch_size):
        self.batch_size = batch_size
        self.test_generator = self.load_data(test_files, self.batch_size)
        self.train_generator = self.load_data(train_files, self.batch_size)
        self.dev_generator = self.load_data(dev_files, self.batch_size)
        self.learning_rate = 4e-4
        self.loss_fn = nn.CrossEntropyLoss()

    @staticmethod
    def load_data(data_files, batch_size):
        df = None
        for data_file in data_files:
            if df is None:
                df = pd.read_csv(data_file)
            else:
                df.append(pd.read_csv(data_file))
        input_ = df['TextEmbeddings'].apply(literal_eval)
        output_ = df['Emotion_Label']
        train = TensorDataset(torch.Tensor(input_), torch.Tensor(output_))
        return DataLoader(train, batch_size=batch_size, shuffle=True)

    @staticmethod
    def train_model(model, loss_fn, optimizer, train_generator, dev_generator):
        """
        Perform the actual training of the model based on the train and dev sets.
        :param model: one of your models_bkp, to be trained to perform 4-way emotion classification
        :param loss_fn: a function that can calculate loss between the predicted and gold labels
        :param optimizer: a created optimizer you will use to update your model weights
        :param train_generator: a DataLoader that provides batches of the training set
        :param dev_generator: a DataLoader that provides batches of the development set
        :return model, the trained model
        """
        epoch = 0
        dev_loss = float('inf')
        while True:
            epoch += 1
            for train_batch in train_generator:
                x_train, y_train = train_batch
                optimizer.zero_grad()
                y_hat_train = model(x_train)
                loss = loss_fn(y_hat_train, y_train.long())
                loss.backward()
                optimizer.step()
            current_dev_loss = 0.0
            with torch.no_grad():
                for dev_batch in dev_generator:
                    x_dev, y_dev = dev_batch
                    y_hat_dev = model(x_dev)
                    current_dev_loss += loss_fn(y_hat_dev, y_dev.long())
                print('Epoch ' + str(epoch) + ' Dev Loss : ' + str(current_dev_loss.item()))
                if current_dev_loss >= dev_loss:
                    break
                dev_loss = current_dev_loss

    @staticmethod
    def test_model(model, loss_fn, test_generator):
        """
        Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
        :param model: a model that performs 4-way emotion classification
        :param loss_fn: a function that can calculate loss between the predicted and gold labels
        :param test_generator: a DataLoader that provides batches of the testing set
        """
        gold = []
        predicted = []

        # Keep track of the loss
        loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
        if USE_CUDA:
            loss = loss.cuda()

        model.eval()

        # Iterate over batches in the test dataset
        with torch.no_grad():
            for X_b, y_b in test_generator:
                # Predict
                y_pred = model(X_b)
                # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
                gold.extend(y_b.cpu().detach().numpy())
                predicted.extend(y_pred.argmax(1).cpu().detach().numpy())
                loss += loss_fn(y_pred.double(), y_b.long()).data

        # Print total loss and macro F1 score
        print("Test loss: ")
        print(loss)
        print("F-score: ")
        print(f1_score(gold, predicted, average='macro'))

    def get_trained_model(self):
        model = EmotionClassifierNetwork()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.train_model(model, self.loss_fn, optimizer, self.train_generator, self.dev_generator)
        return model


def test_meld():
    batch_size = 128
    meld_train = 'meld/data/train_sent_emo_norm_word_embedded.csv'
    meld_test = 'meld/data/test_sent_emo_norm_word_embedded.csv'
    meld_dev = 'meld/data/dev_sent_emo_norm_word_embedded.csv'
    classifier = TextEmotionClassifier([meld_test], [meld_train], [meld_dev], batch_size)
    model = classifier.get_trained_model()


def test_iemocap():
    batch_size = 128
    df_session_1 = pd.read_csv('iemocap/data/Session1_word_embedded.csv')
    df_session_2 = pd.read_csv('iemocap/data/Session2_word_embedded.csv')
    df_session_3 = pd.read_csv('iemocap/data/Session3_word_embedded.csv')
    df_session_4 = pd.read_csv('iemocap/data/Session4_word_embedded.csv')
    df_session_5 = pd.read_csv('iemocap/data/Session5_word_embedded.csv')
    df = pd.concat([df_session_1, df_session_2, df_session_3, df_session_4, df_session_5], ignore_index=True)
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    classifier = TextEmotionClassifier([train], [test], [test], batch_size)
    model = classifier.get_trained_model()


if __name__ == '__main__':
    print('Neural Network Classifier')
    print('MELD Results')
    test_meld()
    print('IEMOCAP Results')
    test_iemocap()
