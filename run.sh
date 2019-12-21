###
# Goals of this script
# 1. Train the BERT Language Model
# 2. Run Unimodal Classifiers (Neural Network and SVM) on the sentence embeddings of the utterance transcriptions
###

TRAIN_BERT=false

# Format the MELD data CSV to normalize the emotion labels and create an ordinal attribute
python meld/meld_dataset_formatter.py

# Format the IEMOCAP data CSV to normalize the emotion labels and create an ordinal attribute
python iemocap/iemocap_dataset_formatter.py

# If BERT training is enabled then perform LM finetuning on MELD data
# Else the code assumes that the models are present in bert/cased and bert/uncased
if [ "$TRAIN_BERT" = true ]; then
  # Prepare MELD data for LM Finetuning
  python meld/prepare_bert_train_data.py

  # Run BERT LM Finetuning - might require a GPU/TPU to execute fast
  python run_lm_finetuning.py --train_data_file='meld/data/language_model/utterances_text.txt' --output_dir='bert/uncased' --do_train --mlm --model_name_or_path="bert-base-uncased"
  python run_lm_finetuning.py --train_data_file='meld/data/language_model/utterances_text.txt' --output_dir='bert/cased' --do_train --mlm --model_name_or_path="bert-base-cased"
fi

# Generate the sentence vectors for MELD and IEMOCAP data
python sentence_embedder.py

# Train a Neural Net and SVM Classifier using the text embeddings
python text_emotion_classifier_nn.py
python text_emotion_classifier_svm.py

# End of Script. Parent script uses the generated multimodal ark files for LDA PLDA classification