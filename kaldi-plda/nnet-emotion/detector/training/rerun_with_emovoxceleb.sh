#!/bin/bash

# author: aa4461, adapted from v2/run.sh

. ./cmd.sh
. ./path.sh
. nnet-emotion/detector/training/finetune_settings.sh
. nnet-emotion/detector/training/emovoxceleb_settings.sh

set -e

stage=0
train_stage=-1
reuse_mfccs=false
include_noise=true
remove_sil=true
min_num_frames=250

# these command line options allow specifying more 
# / fewer layers in the fine-tuned model and different 
# learning rates for the original layers
num_layers=7
first_six_lr=0
dropout=placeholder

. ./utils/parse_options.sh

root="${BASE_DIR}"
mfccdir="$root/mfcc"
vaddir="$root/mfcc"
nnet_dir="$root/exp/xvector_nnet_1a"
data_dir="${DATA_OUTPUT_DIR}"

# make expected directory structure (if it doesn't already exist)
if [ $stage -eq 0 ]; then
  echo "stage 0 (making directory structure): start"

  nnet-emotion/detector/training/prepare_emovoxceleb.sh --stage 0

  echo "stage 0 (making directory structure): end"
fi

# prepare reference model (safe to rerun; it will 
# just over-write any modified reference model)
if [ $stage -eq 1 ]; then
  echo "stage 1 (copying and configuring reference model): start"

  nnet-emotion/detector/training/prepare_emovoxceleb.sh --stage 1 --num_layers $num_layers --first_six_lr $first_six_lr

  echo "stage 1 (copying and configuring reference model): end"
fi

if $reuse_mfccs; then
  echo "reusing already extracted MFCCs..."
  rm -rf ${data_dir}/train_combined
  if $include_noise; then
    # Combine the clean and augmented EmoVoxCeleb list.  This is now roughly double the size of the original clean list.
    utils/combine_data.sh ${data_dir}/train_combined ${data_dir}/train_aug_1m ${DATA_OUTPUT_COMBINED_DIR}
  else 
    # A little hacky but works!  Just moving the un-augmented EmoVoxCeleb where the rest of the code expects it.
    utils/combine_data.sh ${data_dir}/train_combined ${DATA_OUTPUT_COMBINED_DIR}
  fi
else 
  # prepare input data: utt2spk, wav.scp (safe to rerun; it will
  # just over-write any existing generated input files)
  if [ $stage -eq 2 ]; then
    echo "stage 2 (preparing EmoVoxCeleb for Kaldi -- utt2spk, spk2utt): start"
    nnet-emotion/detector/training/prepare_emovoxceleb.sh --stage 2
    echo "stage 2 (preparing EmoVoxCeleb for Kaldi -- utt2spk, spk2utt): end"
  fi

  if [ $stage -eq 3 ]; then
    echo "stage 3 (extracting MFCCs and VAD): start"
    rm -rf $mfccdir
    # Make MFCCs and compute the energy-based VAD for each dataset
    steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
        ${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      ${DATA_OUTPUT_COMBINED_DIR} ${root}/exp/make_vad $vaddir
    utils/fix_data_dir.sh ${DATA_OUTPUT_COMBINED_DIR}
    echo "stage 3 (extracting MFCCs and VAD): end"
  fi

  if $include_noise; then
    # In this section, we augment the EmoVoxCeleb data with reverberation,
    # noise, music, and babble, and combine it with the clean data.
    if [ $stage -eq 4 ]; then
      echo "stage 4 (augmenting with noise): start"
      frame_shift=0.01
      awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' ${DATA_OUTPUT_COMBINED_DIR}/utt2num_frames > ${DATA_OUTPUT_COMBINED_DIR}/reco2dur

      # Make a version with reverberated speech
      rvb_opts=()
      rvb_opts+=(--rir-set-parameters "0.5, $NOISE_DIR/simulated_rirs/smallroom/rir_list")
      rvb_opts+=(--rir-set-parameters "0.5, $NOISE_DIR/simulated_rirs/mediumroom/rir_list")

      # Make a reverberated version of the EmoVoxCeleb list.  Note that we don't add any
      # additive noise here.
      steps/data/reverberate_data_dir.py \
        "${rvb_opts[@]}" \
        --speech-rvb-probability 1 \
        --pointsource-noise-addition-probability 0 \
        --isotropic-noise-addition-probability 0 \
        --num-replications 1 \
        --source-sampling-rate 16000 \
        ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_reverb
      cp ${DATA_OUTPUT_COMBINED_DIR}/vad.scp ${data_dir}/train_reverb/
      utils/copy_data_dir.sh --utt-suffix "-reverb" ${data_dir}/train_reverb ${data_dir}/train_reverb.new
      rm -rf ${data_dir}/train_reverb
      mv ${data_dir}/train_reverb.new ${data_dir}/train_reverb

      # Prepare the MUSAN corpus, which consists of music, speech, and noise
      # suitable for augmentation.
      steps/data/make_musan.sh --sampling-rate 16000 $MUSAN_DIR ${data_dir}

      # Get the duration of the MUSAN recordings.  This will be used by the
      # script augment_data_dir.py.
      for name in speech noise music; do
        utils/data/get_utt2dur.sh ${data_dir}/musan_${name}
        mv ${data_dir}/musan_${name}/utt2dur ${data_dir}/musan_${name}/reco2dur
      done

      # Augment with musan_noise
      steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir ${data_dir}/musan_noise ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_noise
      # Augment with musan_music
      steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir ${data_dir}/musan_music ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_music
      # Augment with musan_speech
      steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir ${data_dir}/musan_speech ${DATA_OUTPUT_COMBINED_DIR} ${data_dir}/train_babble

      # Combine reverb, noise, music, and babble into one directory.
      utils/combine_data.sh ${data_dir}/train_aug ${data_dir}/train_reverb ${data_dir}/train_noise ${data_dir}/train_music ${data_dir}/train_babble
      echo "stage 4 (augmenting with noise): end"
    fi

    if [ $stage -eq 5 ]; then
      echo "stage 5 (sampling and extracting MFCCs for noise): start"
      # Take a random subset of the augmentations
      utils/subset_data_dir.sh ${data_dir}/train_aug 100000 ${data_dir}/train_aug_1m

      utils/fix_data_dir.sh ${data_dir}/train_aug_1m

      # Make MFCCs for the augmented data.  Note that we do not compute a new
      # vad.scp file here.  Instead, we use the vad.scp from the clean version of
      # the list.
      steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
        ${data_dir}/train_aug_1m ${root}/exp/make_mfcc $mfccdir

      # Combine the clean and augmented EmoVoxCeleb list.  This is now roughly
      # double the size of the original clean list.
      utils/combine_data.sh ${data_dir}/train_combined ${data_dir}/train_aug_1m ${DATA_OUTPUT_COMBINED_DIR}
      echo "stage 5 (sampling and extracting MFCCs for noise): end"
    fi
  fi
fi

if [ $stage -eq 6 ]; then 
  echo "stage $stage (adding MELD): start"
  utils/combine_data.sh ${data_dir}/emovoxceleb_and_meld ${data_dir}/train_combined nnet-emotion/meld/outputs/data/all_meld
  rm -rf ${data_dir}/train_combined
  cp -r ${data_dir}/emovoxceleb_and_meld ${data_dir}/train_combined
  echo "end stage $stage"
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -eq 7 ]; then
  echo "stage $stage (removing silence if $remove_sil=True): start"
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  rm -rf ${data_dir}/train_combined_no_sil
  if $remove_sil; then
  	local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
  		${data_dir}/train_combined ${data_dir}/train_combined_no_sil ${root}/exp/train_combined_no_sil
  	utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil
  else 
  	cp -r ${data_dir}/train_combined ${data_dir}/train_combined_no_sil
  fi
  echo "stage $stage (removing silence if $remove_sil=True): end"
fi

# ./run.sh does a bunch of filtering of utterances by speakers
# that are too infrequent -- we skip that here speaker=emotion label 
if [ $stage -eq 8 ]; then
  echo "stage $stage (filtering utterances that are < $min_num_frames): start"
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast $min_num_frames per utterance. (note this is smaller than in v2/run.sh)
  min_len=$min_num_frames
  mv ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' ${data_dir}/train_combined_no_sil/utt2num_frames.bak > ${data_dir}/train_combined_no_sil/utt2num_frames
  utils/filter_scp.pl ${data_dir}/train_combined_no_sil/utt2num_frames ${data_dir}/train_combined_no_sil/utt2spk > ${data_dir}/train_combined_no_sil/utt2spk.new
  mv ${data_dir}/train_combined_no_sil/utt2spk.new ${data_dir}/train_combined_no_sil/utt2spk
  utils/fix_data_dir.sh ${data_dir}/train_combined_no_sil

  echo "stage $stage (filtering utterances that are < $min_num_frames): end"
fi

# safe to rerun, cleans up the directory
if [ $stage -eq 9 ]; then
  echo "stage $stage (getting training examples): start"

  rm -rf $nnet_dir/egs
  sid/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 500000000 \
    --frames-per-iter-diagnostic 100000 \
    --min-frames-per-chunk $min_num_frames \
    --max-frames-per-chunk $min_num_frames \
    --num-diagnostic-archives 3 \
    --num-repeats 500 \
    ${data_dir}/train_combined_no_sil $nnet_dir/egs

  echo "stage $stage (getting training examples): end"
fi

dropout_schedule=$dropout
srand=123
if [ $stage -eq 10 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.input-model "${MODEL_OUTPUT_DIR}/${MODIFIED_REFERENCE_MODEL}" \
    --trainer.optimization.proportional-shrink 10 \
    --trainer.optimization.momentum=0.5 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=1 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=64 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2 \
    --trainer.num-epochs=6 \
    --trainer.dropout-schedule="$dropout_schedule" \
    --trainer.shuffle-buffer-size=1000 \
    --egs.frames-per-eg=1 \
    --egs.dir=$nnet_dir/egs \
    --cleanup.remove-egs false \
    --cleanup.preserve-model-interval=10 \
    --use-gpu=wait \
    --dir=$nnet_dir  || exit 1;
fi
