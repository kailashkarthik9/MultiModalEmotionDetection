# author: aa4461

. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
corpus_path=placeholder
features_path=placeholder

chunk_size=placeholder

. ./utils/parse_options.sh

CSVS_DIR="${corpus_path}/csvs"
DATA_DIR="${corpus_path}/IEMOCAP_full_release"

MFCC_DIR="${features_path}/mfcc"

# prepare the corpus for feature extraction
if [ $stage -eq 0 ]; then
	echo "Stage 0: start"
	for session in 1 2 3 4 5; do
		session_output_path="${features_path}/session${session}"
		mkdir -p $session_output_path
		nnet-emotion/detector/scoring/generate_iemocap_inputs.py \
			"${CSVS_DIR}/Session${session}.csv" \
			"${DATA_DIR}/Session${session}/sentences/wav" \
			$session_output_path
		utils/utt2spk_to_spk2utt.pl "${session_output_path}/utt2spk" > "${session_output_path}/spk2utt"
	done
	utils/combine_data.sh "${features_path}/all_iemocap" "${features_path}/session1" "${features_path}/session2" "${features_path}/session3" "${features_path}/session4" "${features_path}/session5"
	echo "Stage 0: end"
fi

# extract MFCC and pitch features
if [ $stage -eq 1 ]; then
	echo "Stage 1: start"
	steps/make_mfcc_pitch.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --pitch-config conf/pitch.conf --nj 40 --cmd "$train_cmd" \
		"${features_path}/all_iemocap" ${features_path}/exp/make_mfcc $MFCC_DIR
	utils/fix_data_dir.sh "${features_path}/all_iemocap"
	sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      		"${features_path}/all_iemocap" ${features_path}/exp/make_vad $MFCC_DIR
	utils/fix_data_dir.sh "${features_path}/all_iemocap"
	echo "Stage 1: end"
fi

# run extracted features through the nnet
if [ $stage -eq 2 ]; then
	echo "Stage 2: generating predictions for specified model"
	mkdir -p "${model_path}/predictions"
	nnet3-xvector-compute-batched --use-gpu=yes --chunk-size=$chunk_size "${model_path}/final.raw" scp:${features_path}/all_iemocap/feats.scp ark:${model_path}/predictions/iemocap_predictions.ark
	echo "Stage 2: end"
fi

# score nnet predictions against actual labels
if [ $stage -eq 3 ]; then 
	echo "Stage 3: generating scores for specified model"
	mkdir -p "${model_path}/scores"
	nnet-emotion/detector/scoring/score_emotion_prediction_results.py \
		"${features_path}/all_iemocap/utt2spk" \
		"${model_path}/predictions/iemocap_predictions.ark" \
		"${model_path}/scores/iemocap_scores.txt"
	echo "Stage 3: end"
fi
