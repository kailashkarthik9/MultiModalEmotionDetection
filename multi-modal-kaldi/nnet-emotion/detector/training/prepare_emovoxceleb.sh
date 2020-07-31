#!/bin/bash

# author: aa4461

# add compiled Kaldi executables to the path
. ./path.sh
. nnet-emotion/detector/training/finetune_settings.sh
. nnet-emotion/detector/training/emovoxceleb_settings.sh

# hard fail on errors in the script
set -e

# this should be set to our input dimensionality
# (ie # of MFCC + pitch features); it's 30 in the
# default vox2/run.sh training protocol
NUM_INPUT_DIMENSIONS=33
# this should be the number of target emotions
# we're mapping to in EmoVoxCeleb (for our updated output layer)
NUM_TARGET_DIMENSIONS=5

stage=0
# these command line options allow specifying more 
# / fewer layers in the fine-tuned model and different 
# learning rates for the original layers
num_layers=7
first_six_lr=0

. ./utils/parse_options.sh

# make expected directory structure (if it doesn't already exist)
if [ $stage -eq 0 ]; then
	dirs=(
		$MODEL_INPUT_DIR 
		$MODEL_OUTPUT_DIR 
		$DATA_INPUT_DIR 
		$DATA_OUTPUT_COMBINED_DIR
	)
	for dir in "${dirs[@]}"
	do
		if [ ! -d $dir ]; then
			mkdir -p $dir
			chmod -R 775 $dir
		fi
	done
fi

# prepare reference model (safe to rerun; it will
# just over-write any modified reference model)
if [ $stage -eq 1 ]; then
	# first, we delete everything that's already
	# in MODEL_OUTPUT_DIR (so we have a fresh start)
	rm -rf "$MODEL_OUTPUT_DIR/*"

	# then, we generate a new config for our modified
	# model which modifies the output dimensionality of
	# the final layer (from # of speakers in vox2 to # of emotions)
	
	feat_dim=$NUM_INPUT_DIMENSIONS
	num_targets=$NUM_TARGET_DIMENSIONS

	# 
	# START: copied with minimal modification from run_xvector.sh
	#
	max_chunk_size=10000
	min_chunk_size=25

	if [ $num_layers -eq 7 ]; then 
		additional_layers=''
	elif [ $num_layers -eq 8 ]; then
		additional_layers='relu-batchnorm-layer name=tdnn8 dim=512'
	else
		echo "Unsupported num_layers=$num_layers" 1>&2
		exit 1
	fi

	mkdir -p $MODEL_OUTPUT_DIR/configs
	cat <<-EOF > $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig
		# please note that it is important to have input layer with the name=input

		# The frame-level layers
		input dim=${feat_dim} name=input
		relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
		relu-batchnorm-layer name=tdnn2 input=Append(-2,0,2) dim=512
		relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=512
		relu-batchnorm-layer name=tdnn4 dim=512
		relu-batchnorm-layer name=tdnn5 dim=1500

		# The stats pooling layer. Layers after this are segment-level.
		# In the config below, the first and last argument (0, and ${max_chunk_size})
		# means that we pool over an input segment starting at frame 0
		# and ending at frame ${max_chunk_size} or earlier.  The other arguments (1:1)
		# mean that no subsampling is performed.
		stats-layer name=stats config=mean+stddev(0:1:1:${max_chunk_size})

		# This is where we usually extract the embedding (aka xvector) from.
		relu-batchnorm-layer name=tdnn6 dim=512 input=stats

		relu-batchnorm-layer name=tdnn7 dim=512
		${additional_layers}
		output-layer name=output include-log-softmax=true dim=${num_targets}
	EOF

	steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig \
		--config-dir $MODEL_OUTPUT_DIR/configs/
	cp $MODEL_OUTPUT_DIR/configs/final.config $MODEL_OUTPUT_DIR/modified_nnet.config

	echo "$max_chunk_size" > $MODEL_OUTPUT_DIR/max_chunk_size
	echo "$min_chunk_size" > $MODEL_OUTPUT_DIR/min_chunk_size

	#
	# END: copied with minimal modification from run_xvector.sh
	#

	if [ $first_six_lr -lt 0 ]; then 
		echo "Unsupported first_six_lr=$first_six_lr" 1>&2
		exit 1
	fi

	# now, with our updated target config in hand, we copy our
	# reference model, modifying its final output layer and setting
	# the learning rates for the first 6 layers to $first_six_lr
	nnet3-copy \
		--nnet-config="$MODEL_OUTPUT_DIR/modified_nnet.config" \
		--edits="set-learning-rate name=input* learning-rate=$first_six_lr; \
			set-learning-rate name=stats* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn1* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn2* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn3* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn4* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn5* learning-rate=$first_six_lr; \
			set-learning-rate name=tdnn6* learning-rate=$first_six_lr;" \
		"$MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL" \
		"$MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL" || exit 1;
fi

# prepare input data (safe to rerun; it will
# just over-write any existing generated input files)
if [ $stage -eq 2 ]; then
	# make the inputs for the training data
	nnet-emotion/detector/training/generate_emovoxceleb_inputs.py \
		"$DATA_INPUT_DIR" \
		"majority" \
		"$DATA_OUTPUT_COMBINED_DIR"
	utils/utt2spk_to_spk2utt.pl "$DATA_OUTPUT_COMBINED_DIR/utt2spk" > "$DATA_OUTPUT_COMBINED_DIR/spk2utt"
fi
