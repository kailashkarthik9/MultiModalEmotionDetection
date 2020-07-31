import jsonlines
import pandas as pd

EMOTION_IDS = {
    'anger/disgust': 0,
    'fear/surprise': 1,
    'happiness': 2,
    'neutral': 3,
    'sadness': 4
}

DAILY_DIALOG_EMOTION_MAP = {
    'happiness': 'happiness',
    'no_emotion': 'neutral',
    'fear': 'fear/surprise',
    'sadness': 'sadness',
    'disgust': 'anger/disgust',
    'anger': 'anger/disgust',
    'surprise': 'fear/surprise'
}

train_data = list()
with jsonlines.open('train.json') as json_reader:
    for json_object in json_reader:
        for dialog in json_object['dialogue']:
            train_data.append({
                'emotion': dialog['emotion'],
                'act': dialog['act'],
                'text': dialog['text'],
                'topic': json_object['topic']
            })

df_train = pd.DataFrame(train_data)
df_train['label'] = df_train['emotion'].apply(lambda e: EMOTION_IDS[DAILY_DIALOG_EMOTION_MAP[e]])
df_train.to_csv('train.csv')

val_data = list()
with jsonlines.open('valid.json') as json_reader:
    for json_object in json_reader:
        for dialog in json_object['dialogue']:
            val_data.append({
                'emotion': dialog['emotion'],
                'act': dialog['act'],
                'text': dialog['text'],
                'topic': json_object['topic']
            })

df_val = pd.DataFrame(val_data)
df_val['label'] = df_val['emotion'].apply(lambda e: EMOTION_IDS[DAILY_DIALOG_EMOTION_MAP[e]])
df_val.to_csv('val.csv')

test_data = list()
with jsonlines.open('test.json') as json_reader:
    for json_object in json_reader:
        for dialog in json_object['dialogue']:
            test_data.append({
                'emotion': dialog['emotion'],
                'act': dialog['act'],
                'text': dialog['text'],
                'topic': json_object['topic']
            })

df_test = pd.DataFrame(test_data)
df_test['label'] = df_test['emotion'].apply(lambda e: EMOTION_IDS[DAILY_DIALOG_EMOTION_MAP[e]])
df_test.to_csv('test.csv')

print('ok')
