###
# Goals of this script
# 1. Train the BERT Language Model
# 2. Run Unimodal Classifiers (Neural Network and SVM) on the sentence embeddings of the utterance transcriptions
# 3. Use the speech xvectors to create a consolidated multimodal dataset (CSV)
# 4. Create arks with speech, text and multimodal vectors for each utterance to be used for Kaldi LDA-PLDA
###


# Create the required directory structure
mkdir meld/data
mkdir meld/data/language_model
mkdir meld/data/kaldi
mkdir meld/data/kaldi/modified

mkdir iemocap/data
mkdir iemocap/data/kaldi
mkdir iemocap/data/kaldi/modified

###
# Extract MELD and IEMOCAP datasets into meld/data and iemocap/data respectively.
# The data is not committed with the code since they might require permission from the authors to use and distribute
###

# Format the MELD data CSV to normalize the emotion labels and create an ordinal attribute
python meld/meld_dataset_formatter.py

# Format the IEMOCAP data CSV to normalize the emotion labels and create an ordinal attribute
python iemocap/iemocap_dataset_formatter.py

# Prepare MELD data for LM Finetuning
python meld/prepare_bert_train_data.py

# Run BERT LM Finetuning - might require a GPU/TPU to execute fast
mkdir bert
mkdir bert/cased
mkdir bert/uncased
python run_lm_finetuning.py --train_data_file='meld/data/language_model/utterances_text.txt' --output_dir='bert/uncased' --do_train --mlm --model_name_or_path="bert-base-uncased"
python run_lm_finetuning.py --train_data_file='meld/data/language_model/utterances_text.txt' --output_dir='bert/cased' --do_train --mlm --model_name_or_path="bert-base-cased"

# Generate the sentence vectors for MELD and IEMOCAP data
python sentence_embedder.py

# Train a Neural Net and SVM Classifier using the text embeddings
python text_emotion_classifier_nn.py
python text_emotion_classifier_svm.py

# Assume that the .txt files are in meld/data/kaldi and iemocap/data/kaldi
python meld/create_speech_embeddings_csv.py
python iemocap/create_speech_embeddings_csv.py

# Create the multimodal CSVs consolidating the entire dataset into one place for future experiments
python meld/create_combined_embeddings.py
python iemocap/create_combined_embeddings.py

# Create the arks with all three input vectors for LDA - PLDA classification
python meld/create_embeddings_ark.py
sh meld/format_ark_files.sh
python iemocap/create_embeddings_ark.py
sh iemocap/format_ark_files.sh

# End of Script. Parent script uses the generated multimodal ark files for LDA PLDA classification