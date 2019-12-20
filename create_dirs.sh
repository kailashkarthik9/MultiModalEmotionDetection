###
# Goals of this script
# 1. Create the required directory structure for interfacing between kaldi and the python text processing code
###

# Directories for MELD Data
mkdir meld/data
mkdir meld/data/language_model
mkdir meld/data/kaldi
mkdir meld/data/kaldi/modified

# Directories for IEMOCAP Data
mkdir iemocap/data
mkdir iemocap/data/kaldi
mkdir iemocap/data/kaldi/modified

# Directories for BERT Model
mkdir bert
mkdir bert/cased
mkdir bert/uncased

echo 'Please copy and extract the MELD and IEMOCAP datasets into the respective directories. The following code would fail otherwise'

###
# Extract MELD and IEMOCAP datasets into meld/data and iemocap/data respectively.
# The data is not committed with the code since they might require permission from the authors to use and distribute
###