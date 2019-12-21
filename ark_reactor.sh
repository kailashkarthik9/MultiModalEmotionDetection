###
# Goals of this script
# 1. Use the speech xvectors to create a consolidated multimodal dataset (CSV)
# 2. Create arks with speech, text and multimodal vectors for each utterance to be used for Kaldi LDA-PLDA
###

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
