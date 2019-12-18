import glob
import os
import re

import numpy as np
import pandas as pd

attributes = [
    'Sr.No',
    'Utterance',
    'Speaker',
    'Emotion',
    'IDs',
    'StartTime',
    'EndTime',
    'Emotion Label'
]

EMOTION_IDS = {
    'anger/disgust': 0,
    'fear/surprise': 1,
    'happiness': 2,
    'neutral': 3,
    'sadness': 4
}

IEMOCAP_EMOTION_MAP = {
    'hap': 'happiness',
    'sad': 'sadness',
    'ang': 'anger/disgust',
    'fru': 'anger/disgust',
    'exc': 'happiness',
    'fea': 'fear/surprise',
    'sur': 'fear/surprise',
    'dis': 'anger/disgust',
    'neu': 'neutral'
}

SESSIONS = {
    'Session1': 'Ses01',
    'Session2': 'Ses02',
    'Session3': 'Ses03',
    'Session4': 'Ses04',
    'Session5': 'Ses05'
}


class IemocapFormatter:
    # Convert csv to tsv

    @staticmethod
    def extract_utterance_ids(utterance_identifier):
        # Example - Ses01F_impro01_M000 or Ses01F_impro01a_M000 or Ses01F_script01_2_M000 pr Ses05M_script01_1b_F000
        if 'impro' in utterance_identifier:
            if re.search(r'impro.._', utterance_identifier):
                session_number = utterance_identifier[3:5]
                mocap_source = utterance_identifier[5]
                improvisation_number = utterance_identifier[12:14]
                speaker = utterance_identifier[15]
                utterance_number = utterance_identifier[16:]
                return session_number, 'improvisation', improvisation_number, speaker, utterance_number, mocap_source
            else:
                session_number = utterance_identifier[3:5]
                mocap_source = utterance_identifier[5]
                improvisation_number = utterance_identifier[12:15]
                speaker = utterance_identifier[16]
                utterance_number = utterance_identifier[17:]
                return session_number, 'improvisation', improvisation_number, speaker, utterance_number, mocap_source
        else:
            if re.search(r'script.._._', utterance_identifier):
                session_number = utterance_identifier[3:5]
                mocap_source = utterance_identifier[5]
                script_number = utterance_identifier[13:17]
                speaker = utterance_identifier[18]
                utterance_number = utterance_identifier[19:]
                return session_number, 'script', script_number, speaker, utterance_number, mocap_source
            else:
                session_number = utterance_identifier[3:5]
                mocap_source = utterance_identifier[5]
                script_number = utterance_identifier[13:18]
                speaker = utterance_identifier[19]
                utterance_number = utterance_identifier[20:]
                return session_number, 'script', script_number, speaker, utterance_number, mocap_source

    @staticmethod
    def extract_utterance_identifier(session_number, mocap_source, session_type, sub_session_number,
                                     add_utterance_details=False, speaker=None, utterance_number=None):
        file_name = 'Ses' + str(session_number) + str(mocap_source) + '_' + str(session_type) + str(sub_session_number)
        if add_utterance_details:
            file_name += '_' + str(speaker) + str(utterance_number)
        return file_name

    @staticmethod
    def merge_timing_files(male_utterance_directory, female_utterance_directory, target_directory):
        file_paths = glob.glob(male_utterance_directory + '/*')
        for file_path in file_paths:
            male_utterance_file = open(file_path, 'r')
            file_name = os.path.basename(male_utterance_file.name)
            female_utterance_file = open(female_utterance_directory + '/' + file_name, 'r')
            male_utterances = male_utterance_file.read().split('\n')[1:]
            female_utterances = female_utterance_file.read().split('\n')[1:]
            combined_utterances = []
            for utterance in male_utterances:
                if not utterance:
                    continue
                utterance_splits = utterance.split()
                combined_utterances.append(
                    utterance_splits[2] + ', ' + utterance_splits[0] + ', ' + utterance_splits[1])
            for utterance in female_utterances:
                if not utterance:
                    continue
                utterance_splits = utterance.split()
                combined_utterances.append(
                    utterance_splits[2] + ', ' + utterance_splits[0] + ', ' + utterance_splits[1])
            target_file = open(target_directory + '/' + file_name[:-3] + 'csv', 'w')
            target_file.write('\n'.join(combined_utterances))
            target_file.close()
            male_utterance_file.close()
            female_utterance_file.close()

    @staticmethod
    def convert_transcripts_to_csv(directory, target_directory):
        file_paths = glob.glob(directory + '/*')
        for file_path in file_paths:
            if os.path.isdir(file_path):
                continue
            file = open(file_path, 'r')
            file_name = os.path.basename(file.name)
            utterances = file.read().split('\n')
            formatted_utterances = []
            for utterance in utterances:
                if not utterance:
                    continue
                identifier_end_idx = utterance.find(' ')
                if identifier_end_idx == -1:
                    continue
                time_end_idx = utterance.find(' ', identifier_end_idx + 1)
                if time_end_idx == -1:
                    continue
                utterance_splits = [utterance[:identifier_end_idx], utterance[identifier_end_idx + 1:time_end_idx],
                                    utterance[time_end_idx + 1:]]
                time_splits = utterance_splits[1][1:-2].split('-')
                if len(time_splits) != 2:
                    continue
                formatted_utterances.append(
                    utterance_splits[0] + '\t' + utterance_splits[2].replace('\t', ' ') + '\t' + time_splits[0] + '\t' +
                    time_splits[1])
            target_file = open(target_directory + '/' + file_name[:-3] + 'tsv', 'w')
            target_file.write('\n'.join(formatted_utterances))
            target_file.close()
            file.close()

    @staticmethod
    def convert_labels_to_csv(directory, target_directory):
        file_paths = glob.glob(directory + '/*')
        for file_path in file_paths:
            if os.path.isdir(file_path):
                continue
            file = open(file_path, 'r')
            file_name = os.path.basename(file.name)
            utterances = file.read().split('\n\n')[1:]
            formatted_utterances = []
            for utterance in utterances:
                if not utterance:
                    continue
                utterance_first_line = utterance.split('\n')[0]
                identifier_start_idx = utterance_first_line.find(']')
                vad_start_idx = utterance_first_line.find('[', identifier_start_idx + 1)
                utterance_sliced = utterance_first_line[identifier_start_idx + 2: vad_start_idx - 1]
                utterance_splits = utterance_sliced.split()
                if len(utterance_splits) != 2:
                    raise Exception
                formatted_utterances.append(utterance_splits[0] + ', ' + utterance_splits[1])
            target_file = open(target_directory + '/' + file_name[:-3] + 'csv', 'w')
            target_file.write('\n'.join(formatted_utterances))
            target_file.close()
            file.close()

    @staticmethod
    def aggregate_data(transcripts_directory, timings_directory, labels_directory, target_file):
        aggregated_data = None
        transcript_file_paths = glob.glob(transcripts_directory + '/*')
        for transcript_file_path in transcript_file_paths:
            if os.path.isdir(transcript_file_path):
                continue
            file_name = os.path.basename(transcript_file_path)[:-4]
            print(file_name)
            transcript = pd.read_csv(transcript_file_path, sep='\t', header=None)
            labels = pd.read_csv(labels_directory + '/' + file_name + '.csv', header=None)
            merged_data = transcript.merge(labels, on=0)
            merged_data.columns = ['Identifier', 'Utterance', 'StartTime', 'EndTime', 'Emotion']
            merged_data['Emotion'] = merged_data['Emotion'].apply(lambda x: x.strip())
            merged_data = merged_data[merged_data.Emotion.isin(IEMOCAP_EMOTION_MAP.keys())]
            merged_data['Emotion_Label'] = merged_data['Emotion'].apply(
                lambda x: EMOTION_IDS[IEMOCAP_EMOTION_MAP[x.strip()]])
            merged_data['Session_Number'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[0])
            merged_data['Dialogue_Type'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[1])
            merged_data['Dialogue_Number'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[2])
            merged_data['Speaker'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[3])
            merged_data['Utterance_Number'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[4])
            merged_data['Mocap_Source'] = merged_data['Identifier'].apply(
                lambda x: IemocapFormatter.extract_utterance_ids(x)[5])
            merged_data = merged_data.drop(columns=['Identifier'])
            if aggregated_data is None:
                aggregated_data = merged_data
            else:
                aggregated_data = aggregated_data.append(merged_data)
        aggregated_data.index = np.arange(1, len(aggregated_data) + 1)
        aggregated_data = aggregated_data[
            ['Session_Number', 'Mocap_Source', 'Dialogue_Type', 'Dialogue_Number', 'Utterance_Number', 'StartTime',
             'EndTime', 'Utterance', 'Speaker', 'Emotion', 'Emotion_Label']]
        aggregated_data.to_csv(target_file, index_label='Sr.No')


if __name__ == '__main__':
    for session_lf, session_sf in SESSIONS.items():
        # IemocapFormatter.merge_timing_files('data/' + session_lf + '/dialog/lab/' + session_sf + '_M',
        #                                     'data/' + session_lf + '/dialog/lab/' + session_sf + '_F',
        #                                     'data/' + session_lf + '/dialog/lab')
        # os.makedirs('data/' + session_lf + '/dialog/transcriptions/tsv', exist_ok=True)
        # IemocapFormatter.convert_transcripts_to_csv('data/' + session_lf + '/dialog/transcriptions', 'data/' + session_lf + '/dialog/transcriptions/tsv')
        # os.makedirs('data/' + session_lf + '/dialog/EmoEvaluation/csv', exist_ok=True)
        # IemocapFormatter.convert_labels_to_csv('data/' + session_lf + '/dialog/EmoEvaluation', 'data/' + session_lf + '/dialog/EmoEvaluation/csv')
        IemocapFormatter.aggregate_data('data/' + session_lf + '/dialog/transcriptions/tsv',
                                        'data/' + session_lf + '/dialog/lab',
                                        'data/' + session_lf + '/dialog/EmoEvaluation/csv',
                                        'data/' + session_lf + '.csv')
