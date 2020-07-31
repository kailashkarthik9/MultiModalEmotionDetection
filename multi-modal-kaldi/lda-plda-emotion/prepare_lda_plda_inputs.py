import pickle
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from kaldiio import ReadHelper, WriteHelper

CREMA_D_PATH = 'nnet-emotion/cremad/outputs/data/all_cremad/'
MELD_PATH = 'nnet-emotion/meld/outputs/data/all_meld/'
IEMOCAP_PATH = 'nnet-emotion/iemocap/all_iemocap/'

VALID_TRAINING_CORPORA = set([
    'cremad',
    'meld',
    *['iemocap%s' % i for i in range(1, 6)]
])

EMOTION_TO_ID = {emotion: id for id, emotion in
                 enumerate(['anger/disgust', 'fear/surprise', 'happiness', 'neutral', 'sadness'])}


# example utterance id: anger/disgust-1001_DFA_ANG_XX
def get_cremad_utterances(speech_dir, text_dir):
    utterances = defaultdict(dict)
    with open('%s/utt2spk' % (CREMA_D_PATH), 'r') as f:
        for line in f.readlines():
            utterance_id, _ = line.split(' ')

            if utterance_id in utterances:
                raise Exception("Duplicate utterance: %s" % utterance_id)

            utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

    if speech_dir != 'none':
        with ReadHelper('scp:%s/cremad/xvector.scp' % speech_dir) as reader:
            for utterance_id, speech_vector in reader:
                if utterance_id not in utterances:
                    raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

                utterances[utterance_id]['speech'] = speech_vector

    if text_dir != 'none':
        embeddings_by_emotion = defaultdict(list)
        with open('%s/dd_embedding.pkl' % text_dir, 'rb') as f:
            dd = pickle.load(f)
            for _, row in dd.iterrows():
                utterance_id = row['ID']
                emotion = utterance_id.split('-')[0]
                embeddings_by_emotion[emotion].append(row['Text Embeddings'])

        for utterance_id, utterance in utterances.items():
            emotion = utterance['emotion']
            random.seed(utterance_id)
            random_text_vector = random.choice(embeddings_by_emotion[emotion])
            utterance['text'] = random_text_vector
    return utterances


# example utterance id: anger/disgust-dev-1-11
def get_meld_utterances(speech_dir, text_dir):
    utterances = defaultdict(dict)
    with open('%s/utt2spk' % (MELD_PATH), 'r') as f:
        for line in f.readlines():
            utterance_id, _ = line.split(' ')

            if utterance_id in utterances:
                raise Exception("Duplicate utterance: %s" % utterance_id)

            utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

    if speech_dir != 'none':
        with ReadHelper('scp:%s/meld/xvector.scp' % speech_dir) as reader:
            for utterance_id, speech_vector in reader:
                if utterance_id not in utterances:
                    raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

                utterances[utterance_id]['speech'] = speech_vector

    if text_dir != 'none':
        with open('%s/meld_embedding.pkl' % text_dir, 'rb') as f:
            meld = pickle.load(f)
            for _, row in meld.iterrows():
                utterance_id, text_embeddings = row['ID'], row['Text Embeddings']

                if utterance_id not in utterances:
                    raise Exception("Text vector for unknown utterance: %s" % utterance_id)

                utterances[utterance_id]['text'] = text_embeddings
    return utterances


# example utterance id: sadness-test-05M-script-02_2-032-M
def get_iemocap_utterances(speech_dir, text_dir, subset):
    utterances = defaultdict(dict)
    with open('%s/utt2spk' % (IEMOCAP_PATH), 'r') as f:
        for line in f.readlines():
            utterance_id, _ = line.split(' ')
            session = int(utterance_id.split('-')[2][0:2])

            if session != int(subset[-1:]):
                continue

            if utterance_id in utterances:
                raise Exception("Duplicate utterance: %s" % utterance_id)

            utterances[utterance_id]['session'] = session
            utterances[utterance_id]['emotion'] = utterance_id.split('-')[0]

    if speech_dir != 'none':
        with ReadHelper('scp:%s/iemocap/xvector.scp' % speech_dir) as reader:
            for utterance_id, speech_vector in reader:
                session = int(utterance_id.split('-')[2][0:2])

                if session != int(subset[-1:]):
                    continue

                if utterance_id not in utterances:
                    raise Exception("Speech vector for unknown utterance: %s" % utterance_id)

                utterances[utterance_id]['speech'] = speech_vector

    if text_dir != 'none':
        with open('%s/iemocap_embedding.pkl' % text_dir, 'rb') as f:
            iemocap = pickle.load(f)
            for _, row in iemocap.iterrows():
                utterance_id = row['ID']
                session = int(utterance_id.split('-')[2][0:2])

                if session != int(subset[-1:]):
                    continue

                if utterance_id not in utterances:
                    raise Exception("Text vector for unknown utterance: %s" % utterance_id)

                utterances[utterance_id]['text'] = row['Text Embeddings']

    return utterances


def get_corpus_utterances(speech_dir, text_dir, corpus):
    if corpus == 'cremad':
        return get_cremad_utterances(speech_dir, text_dir)
    elif corpus == 'meld':
        return get_meld_utterances(speech_dir, text_dir)
    else:
        return get_iemocap_utterances(speech_dir, text_dir, corpus)


def get_utterances(speech_dir, text_dir, corpora):
    utterances = {}
    for corpus in corpora:
        for utterance_id, utterance in get_corpus_utterances(speech_dir, text_dir, corpus).items():
            if utterance_id in utterances:
                raise Exception("Duplicate utterance: %s" % utterance_id)
            utterances[utterance_id] = utterance
    return utterances


def write_output_files(prefix, utterances, has_speech, has_text, output_dir):
    # utt2spk -- index
    # spk2utt -- index
    # xvector.scp, xvector.ark

    utt2spk = {}
    spk2utt = defaultdict(list)
    for utterance_id, utterance in utterances.items():
        emotion = EMOTION_TO_ID[utterance['emotion']]

        utt2spk[utterance_id] = emotion
        spk2utt[emotion].append(utterance_id)

    with open('%s/%s_utt2spk' % (output_dir, prefix), 'w') as f:
        for utterance_id in sorted(utt2spk.keys()):
            f.write("%s %s\n" % (utterance_id, utt2spk[utterance_id]))

    with open('%s/%s_spk2utt' % (output_dir, prefix), 'w') as f:
        for speaker in sorted(spk2utt.keys()):
            f.write("%s %s\n" % (speaker, ' '.join(spk2utt[speaker])))

    dimensions = Counter()
    with WriteHelper(
            'ark,scp:%s/%s_xvector.ark,%s/%s_xvector.scp' % (
                    output_dir, prefix, output_dir, prefix)) as writer:
        for utterance_id in sorted(utterances.keys()):
            utterance = utterances[utterance_id]
            if has_speech and not has_text:
                if 'speech' not in utterance:
                    print("Missing speech vector: %s" % utterance_id)
                    continue
                feature_vector = utterance['speech']
            elif has_text and not has_speech:
                feature_vector = utterance['text']
            else:
                if 'speech' not in utterance:
                    print("Missing speech vector: %s" % utterance_id)
                    continue
                feature_vector = np.concatenate((
                    utterances[utterance_id]['speech'],
                    utterances[utterance_id]['text']
                ))
            dimensions[feature_vector.shape] += 1
            writer(
                utterance_id,
                feature_vector
            )
    print("%s dimensions" % prefix)
    print(dimensions)


def write_trials_file(train_utterances, test_utterances, output_dir):
    num_target = 0
    num_nontarget = 0
    with open('%s/trials' % output_dir, 'w') as f:
        for train_utterance_id in sorted(train_utterances.keys()):
            for test_utterance_id in sorted(test_utterances.keys()):
                assert not train_utterance_id == test_utterance_id

                if train_utterances[train_utterance_id]['emotion'] == test_utterances[test_utterance_id]['emotion']:
                    label = 'target'
                    num_target += 1
                else:
                    label = 'nontarget'
                    num_nontarget += 1
                f.write('%s %s %s\n' % (train_utterance_id, test_utterance_id, label))
    print("target: %s, nontarget: %s" % (num_target, num_nontarget))


speech_dir = sys.argv[1]
text_dir = sys.argv[2]
train_corpora = set(sys.argv[3].split(','))
output_dir = sys.argv[4]

##
# validate specified training corpora
##

invalid_corpora = train_corpora - VALID_TRAINING_CORPORA
if len(invalid_corpora) > 0:
    raise Exception("Unknown corpora: %s" % (invalid_corpora))

iemocap_subset_to_exclude = list(filter(lambda corpus: 'iemocap' in corpus, train_corpora))
if len(iemocap_subset_to_exclude) > 1:
    raise Exception("Can't specify excluding more than 1 IEMOCAP subset: %s" % (iemocap_subset_to_exclude))
elif len(iemocap_subset_to_exclude) == 1:
    train_corpora.remove(iemocap_subset_to_exclude[0])
    for iemocap_subset in ['iemocap%s' % i for i in range(1, 6)]:
        if iemocap_subset != iemocap_subset_to_exclude[0]:
            train_corpora.add(iemocap_subset)

test_corpora = ['iemocap%s' % i for i in range(1, 6)] \
    if len(iemocap_subset_to_exclude) == 0 else iemocap_subset_to_exclude

print("Train corpora: %s" % train_corpora)
print("Test corpora: %s" % test_corpora)

train_utterances = get_utterances(speech_dir, text_dir, train_corpora)
test_utterances = get_utterances(speech_dir, text_dir, test_corpora)

write_output_files('train', train_utterances, speech_dir != 'none', text_dir != 'none', output_dir)
write_output_files('test', test_utterances, speech_dir != 'none', text_dir != 'none', output_dir)
write_trials_file(train_utterances, test_utterances, output_dir)
