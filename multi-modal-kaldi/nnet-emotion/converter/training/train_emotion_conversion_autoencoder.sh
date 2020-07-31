#!/bin/bash

# author: aa4461

# add compiled Kaldi executables to the path
. ./cmd.sh
. ./path.sh
. nnet-emotion/converter/autoencoder_settings.sh

# hard fail on errors in the script
set -e

stage=placeholder
train_stage=-1

discriminator_model="seven_layers_with_sil_250"
encoder_architecture=placeholder

. ./utils/parse_options.sh

# this should be set to our underlying feature dimensionality
# (ie # of MFCC + pitch features, so 33); this will be the 
# dimensionality of our encoded frames in latent space
NUM_FEAT_DIMENSIONS=33

# this should be the number of target emotions
# we're mapping to in MELD (for our updated output layer)
NUM_TARGET_DIMENSIONS=5

# the minimum utterance length (in frames) that we'll
# train the autoencoder on; for now, we keep it in sync
# with whatever the discriminator was trained on
MIN_UTT_LENGTH=250

# set up expected input directory structure,
# copy reference model and source data for training sets
if [ $stage -le 0 ]; then
	echo "STAGE 0 START: setting up directory structure, copying input model and data"

	dirs=(
		$BASE_DIR
		$MODEL_INPUT_DIR 
		$MODEL_OUTPUT_DIR 
		$DATA_INPUT_DIR 
		$DATA_OUTPUT_DIR
	)
	for dir in "${dirs[@]}"
	do
		if [ ! -d $dir ]; then
			mkdir -p $dir
			sudo chmod -R 775 $dir
		fi
	done

	# copy the specified discriminator into our MODEL_INPUT_DIR
	cp ../../../../pretrained/nnet/models/${discriminator_model}.raw $MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL

	# copy the MELD and IEMOCAP features, labels, predictions for the specified discriminator
	cp nnet-emotion/meld/outputs/data/all_meld/wav.scp $DATA_INPUT_DIR/meld_wav.scp
	cp nnet-emotion/meld/outputs/data/all_meld/utt2spk $DATA_INPUT_DIR/meld_utt2spk
	cp ../../../../pretrained/nnet/predictions/meld/${discriminator_model}_prediction.ark $DATA_INPUT_DIR/meld_predictions.ark

	cp nnet-emotion/iemocap/all_iemocap/wav.scp $DATA_INPUT_DIR/iemocap_wav.scp
	cp nnet-emotion/iemocap/all_iemocap/utt2spk $DATA_INPUT_DIR/iemocap_utt2spk
	cp ../../../../pretrained/nnet/predictions/iemocap/${discriminator_model}_prediction.ark $DATA_INPUT_DIR/iemocap_predictions.ark

	echo "STAGE 0 END: setting up directory structure, copying input model and data"
fi

# convert the reference model (the trained emotion detector) into an autoencoder 
# (we use the emotion detector as its final pinned layers to guide training)
if [ $stage -le 1 ]; then
	echo "STAGE 1 START: converting emotion detector into autoencoder"

	# first, we delete everything that's already
	# in MODEL_OUTPUT_DIR (so we have a fresh start)
	rm -rf "$MODEL_OUTPUT_DIR/*"

	# then, we generate a new config for our modified
	# model which add a bunch of layers for the autoencoder
	
	input_dim=$(($NUM_FEAT_DIMENSIONS + 1))
	latent_dim=$NUM_FEAT_DIMENSIONS
	output_dim=$NUM_TARGET_DIMENSIONS

	max_chunk_size=10000
	min_chunk_size=25
	mkdir -p $MODEL_OUTPUT_DIR/configs
	if [ $encoder_architecture = "CNN" ]; then
		cat <<-EOF > $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig
			# please note that it is important to have input layer with the name=input

			# autoencoder layers
			input dim=${input_dim} name=input
			relu-batchnorm-layer name=tdnn-3 dim=1024 input=Append(-2,-1,0,1,2)
			relu-batchnorm-layer name=tdnn-2 dim=512 input=Append(-1,2)
			relu-batchnorm-layer name=tdnn-1 dim=${latent_dim} input=Append(-3,3)

			# below are the layers from our pre-trained emotion discriminator
			# (left unchanged so we reuse their weights -- learning rates are 0 below)
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
			relu-batchnorm-layer name=tdnn6 dim=512 input=stats
			relu-batchnorm-layer name=tdnn7 dim=512

			output-layer name=output include-log-softmax=true dim=${output_dim}
		EOF
	elif [ $encoder_architecture = "FF" ]; then
		cat <<-EOF > $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig
			# please note that it is important to have input layer with the name=input

			# autoencoder layers
			input dim=${input_dim} name=input
			relu-batchnorm-layer name=tdnn-3 dim=${latent_dim} 
			relu-batchnorm-layer name=tdnn-2 dim=${latent_dim} 
			relu-batchnorm-layer name=tdnn-1 dim=${latent_dim} 

			# below are the layers from our pre-trained emotion discriminator
			# (left unchanged so we reuse their weights -- learning rates are 0 below)
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
			relu-batchnorm-layer name=tdnn6 dim=512 input=stats
			relu-batchnorm-layer name=tdnn7 dim=512

			output-layer name=output include-log-softmax=true dim=${output_dim}
		EOF
	else 
		echo "Unsupported encoder architecture: $encoder_architecture" 1>&2
		exit 1
	fi

	steps/nnet3/xconfig_to_configs.py \
		--xconfig-file $MODEL_OUTPUT_DIR/configs/modified_nnet.xconfig \
		--config-dir $MODEL_OUTPUT_DIR/configs/
	cp $MODEL_OUTPUT_DIR/configs/final.config $MODEL_OUTPUT_DIR/modified_nnet.config

	# we'll use this to extract emotion-converted frames from the model
	echo "output-node name=output input=tdnn-1.affine" > $MODEL_OUTPUT_DIR/extract.config

	# now, with our updated target config in hand, we copy our
	# reference model, modifying its final output layer and setting
	# the learning rates for the first 6 layers to 0
	nnet3-copy \
		--nnet-config="$MODEL_OUTPUT_DIR/modified_nnet.config" \
		--edits="
			set-learning-rate name=stats* learning-rate=0; \
			set-learning-rate name=tdnn1* learning-rate=0; \
			set-learning-rate name=tdnn2* learning-rate=0; \
			set-learning-rate name=tdnn3* learning-rate=0; \
			set-learning-rate name=tdnn4* learning-rate=0; \
			set-learning-rate name=tdnn5* learning-rate=0; \
			set-learning-rate name=tdnn6* learning-rate=0; \
			set-learning-rate name=tdnn7* learning-rate=0; \
			set-learning-rate name=output* learning-rate=0;" \
		"$MODEL_INPUT_DIR/$BASE_REFERENCE_MODEL" \
		"$MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL" || exit 1;

	nnet3-info $MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL	

	echo "STAGE 1 END: converting emotion detector into autoencoder"
fi

# select training examples from MELD and IEMOCAP based on
# how well the detector classifies them (choose only high accuracy examples)
if [ $stage -le 2 ]; then
	echo "STAGE 2 START: selecting training examples from MELD/IEMOCAP"

	nnet-emotion/converter/training/generate_emotion_conversion_inputs.py $DATA_INPUT_DIR $DATA_OUTPUT_DIR
	utils/utt2spk_to_spk2utt.pl $DATA_OUTPUT_DIR/utt2spk > $DATA_OUTPUT_DIR/spk2utt

	echo "STAGE 2 END: selecting training examples from MELD/IEMOCAP"
fi

# generate MFCC and pitch features for our training examples
if [ $stage -le 3 ]; then
	echo "STAGE 3 START: generating MFCC and pitch features"

	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		$DATA_OUTPUT_DIR ${BASE_DIR}/exp/make_mfcc ${BASE_DIR}/mfcc
	utils/fix_data_dir.sh $DATA_OUTPUT_DIR

  	echo "STAGE 3 END: generating MFCC and pitch features"
fi

# concatenate the target emotion as the 34th feature for each of our training examples
if [ $stage -le 4 ]; then
	echo "STAGE 4 START: concatenating target emotion as 34th feature"

	expanded_feature_dir=${BASE_DIR}/mfcc_and_target
	rm -rf $expanded_feature_dir
	mkdir -p $expanded_feature_dir

	nnet-emotion/converter/training/expand_mfccs_and_pitch_features_with_target_emotion.py ${BASE_DIR}/mfcc $expanded_feature_dir

	cat ${BASE_DIR}/mfcc_and_target/*scp > $DATA_OUTPUT_DIR/feats.scp

	echo "STAGE 4 END: concatenating target emotion as 34th feature"
fi

# filter out utterances that are too short
if [ $stage -le 5 ]; then
	echo "STAGE 5 START: filtering out utterances that are too short"

	mv $DATA_OUTPUT_DIR/utt2num_frames $DATA_OUTPUT_DIR/utt2num_frames.bak
	awk -v min_len=$MIN_UTT_LENGTH '$2 > min_len {print $1, $2}' $DATA_OUTPUT_DIR/utt2num_frames.bak > $DATA_OUTPUT_DIR/utt2num_frames
	utils/filter_scp.pl $DATA_OUTPUT_DIR/utt2num_frames $DATA_OUTPUT_DIR/utt2spk > $DATA_OUTPUT_DIR/utt2spk.new
	mv $DATA_OUTPUT_DIR/utt2spk.new $DATA_OUTPUT_DIR/utt2spk
	utils/fix_data_dir.sh $DATA_OUTPUT_DIR

	echo "STAGE 5 END: filtering out utterances that are too short"
fi

# generate training egs from training featureset
if [ $stage -le 6 ]; then
	echo "STAGE 6 START: generating egs for neural net training"

	sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
	    --nj 8 \
	    --stage 0 \
	    --frames-per-iter 75000000 \
	    --frames-per-iter-diagnostic 100000 \
	    --min-frames-per-chunk 50 \
	    --max-frames-per-chunk $MIN_UTT_LENGTH \
	    --num-diagnostic-archives 3 \
	    --num-repeats 500 \
	    $DATA_OUTPUT_DIR $BASE_DIR/egs

	echo "STAGE 6 END: generating egs for neural net training"
fi

# train the autoencoder
if [ $stage -le 7 ]; then
	echo "STAGE 7 START: training neural net!"

	mkdir -p $BASE_DIR/nnet

	dropout_schedule='0,0@0.20,0.1@0.50,0'
	srand=123
	steps/nnet3/train_raw_dnn.py --stage=$train_stage \
	    --cmd="$train_cmd" \
	    --trainer.input-model $MODEL_OUTPUT_DIR/$MODIFIED_REFERENCE_MODEL \
	    --trainer.optimization.proportional-shrink 10 \
	    --trainer.optimization.momentum=0.5 \
	    --trainer.optimization.num-jobs-initial=1 \
	    --trainer.optimization.num-jobs-final=3 \
	    --trainer.optimization.initial-effective-lrate=0.001 \
	    --trainer.optimization.final-effective-lrate=0.0001 \
	    --trainer.optimization.minibatch-size=64 \
	    --trainer.srand=$srand \
	    --trainer.max-param-change=2 \
	    --trainer.num-epochs=3 \
	    --trainer.dropout-schedule="$dropout_schedule" \
	    --trainer.shuffle-buffer-size=1000 \
	    --egs.frames-per-eg=1 \
	    --egs.dir=$BASE_DIR/egs \
	    --cleanup.remove-egs false \
	    --cleanup.preserve-model-interval=10 \
	    --use-gpu=wait \
	    --dir=$BASE_DIR/nnet  || exit 1;

	echo "STAGE 7 END: training neural net!"
fi

# copy the autoencoder, remove output layers for converter
if [ $stage -le 8 ]; then
	echo "STAGE 8 START: copying model and removing detector"

	mkdir -p $BASE_DIR/models
	cp $BASE_DIR/nnet/final.raw $BASE_DIR/models/${encoder_architecture}_full.raw

	nnet3-copy \
		--nnet-config="$MODEL_OUTPUT_DIR/extract.config" \
		"${BASE_DIR}/models/${encoder_architecture}_full.raw" \
		"${BASE_DIR}/models/${encoder_architecture}_converter.raw"

	echo "STAGE 8 END: copying model and removing detector"
fi
