corpora = [
    'Crema-D Embeddings',
    'DailyDialog Embeddings',
    'IEMOCAP Embeddings',
    'MELD Embeddings'
]

corpora_map = {
    'Crema-D Embeddings': 'crema',
    'DailyDialog Embeddings': 'dd',
    'IEMOCAP Embeddings': 'iemocap',
    'MELD Embeddings': 'meld'
}

models = [
    'DailyDialog',
    'DailyDialog+MELD',
    'DailyDialog+IEMOCAP/0',
    'DailyDialog+IEMOCAP/1',
    'DailyDialog+IEMOCAP/2',
    'DailyDialog+IEMOCAP/3',
    'DailyDialog+IEMOCAP/4',
    'DailyDialog+MELD+IEMOCAP/0',
    'DailyDialog+MELD+IEMOCAP/1',
    'DailyDialog+MELD+IEMOCAP/2',
    'DailyDialog+MELD+IEMOCAP/3',
    'DailyDialog+MELD+IEMOCAP/4',
]

from os import path, mkdir
from shutil import copyfile

output_dir = 'output'
if not path.isdir(output_dir):
    mkdir(output_dir)

for model in models:
    for corpus in corpora:
        input_path = path.join('Embeddings', corpus, model, 'text_embeddings.pkl')
        output_folder_path = path.join(output_dir, model.replace('/', '_'))
        if not path.isdir(output_folder_path):
            mkdir(output_folder_path)
        output_path = path.join(output_folder_path, corpora_map[corpus] + '_embedding.pkl')
        copyfile(input_path, output_path)
