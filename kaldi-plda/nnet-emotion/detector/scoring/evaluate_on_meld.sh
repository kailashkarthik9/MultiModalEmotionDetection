# author: aa4461

. ./cmd.sh
. ./path.sh

set -e

stage=placeholder
model_path=placeholder
meld_path=placeholder
output_path=placeholder

. ./utils/parse_options.sh

BASE_DIR="${meld_path}"

# run extracted features through the nnet
if [ $stage -eq -1 ]; then
	echo "Stage -1: start"
	# we begin by splitting the meld dataset
	# so we can process the utterances 1-by-1
	# (we do this because some of the utterances
	# cause nnet3-compute-batch to hang....)
	calling_directory=`pwd`	
	split_directory=$meld_path/outputs/data/all_meld/splits
	rm -rf $split_directory
	mkdir -p $split_directory
	cd $split_directory
	split ../feats.scp -l 100 split
	
	cd $calling_directory
	echo "Stage -1: end"
fi

if [ $stage -eq 0 ]; then
	echo "Stage 0: start"
	mkdir -p $output_path/predictions
	for layers in seven_layers eight_layers; do
		for mode in no_sil with_sil; do
			for min_frame_len in 100 150 200 250 300; do 
				model="${layers}_${mode}_${min_frame_len}"
				if [ ! -f "${model_path}/${model}.raw" ]
				then
					echo "${model} does not exist!"
					continue
				fi
				if [ -f "${output_path}/predictions/${model}_prediction.ark" ]
				then
					echo "${model} predictions exist, skipping"
					continue
				fi
				for split in `ls ${meld_path}/outputs/data/all_meld/splits`; do
					(timeout 10s nnet3-compute-batch --use-gpu=wait "${model_path}/${model}.raw" scp:${meld_path}/outputs/data/all_meld/splits/${split} ark:${output_path}/predictions/${model}_${split}_prediction.ark || echo "$split timed out!") &>> ${output_path}/predictions/meld_${model}_predictions_log
				done
			done
		done
	done
	echo "Stage 0: end"
fi

if [ $stage -eq 1 ]; then
	echo "Stage 1: start"
	for layers in seven_layers eight_layers; do
		for mode in no_sil with_sil; do
			for min_frame_len in 100 150 200 250 300; do
				model="${layers}_${mode}_${min_frame_len}"
				splits=(`ls ${output_path}/predictions/${model}* | grep -v "splitad" | grep -v "splitbs"`)
				merge_emotion_prediction_results.py "${splits[@]}" ${output_path}/predictions/${model}_prediction.ark
			done
		done
	done
	echo "Stage 2: end"
fi

# score nnet predictions against actual labels
if [ $stage -eq 2 ]; then
	echo "Stage 2: start"
	mkdir -p "${output_path}/scores"
	for layers in seven_layers eight_layers; do
		for mode in no_sil with_sil; do
			for min_frame_len in 100 150 200 250 300; do
				model="${layers}_${mode}_${min_frame_len}"
				score_emotion_prediction_results.py \
					$model \
					"${output_path}/outputs/data/all_meld/utt2spk" \
					"${output_path}/predictions/${model}_prediction.ark" \
					"${output_path}/scores"
			done
		done
	done
	echo "Stage 2: end"
fi
