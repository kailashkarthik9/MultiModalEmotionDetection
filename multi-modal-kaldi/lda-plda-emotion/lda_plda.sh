. ./cmd.sh
. ./path.sh

set -e

# speech xvectors: variants = layers 6, 7
# text xvectors: variants = DD, DD+IEMOCAP, DD+MELD, DD+MELD+IEMOCAP
#	download from: https://drive.google.com/drive/u/1/folders/1CypiTlTcYxoP93leThBjvUPlH1I5nK9-
# corpora: 
#	CremaD/DD 
#	MELD
# 	CremaD/DD, MELD
#	CremaD/DD, 4/5 of IEMOCAP
#	MELD, 4/5 of IEMOCAP
# 	CremaD/DD, MELD, 4/5 of IEMOCAP

# questions:
#	should we be subtracting corpus-level means?
# 	they use the mean vectors for the "train" vectors for EER, should we?
#		(fwiw they don't for voxceleb)

variant=placeholder
speech_dir=placeholder
text_dir=placeholder
train_corpora=placeholder
output_dir=placeholder

stage=0
lda_dim=200
work_dir="lda_plda_work"

. parse_options.sh || exit 1;

echo "Starting from stage $stage..."

if [ $stage -le 0 ]; then
	rm -rf $work_dir
	mkdir -p $work_dir
fi

if [ $stage -le 1 ]; then
	echo "Preparing training and testing inputs..."
	python lda-plda-emotion/prepare_lda_plda_inputs.py $speech_dir $text_dir $train_corpora $work_dir
fi

if [ $stage -le 2 ]; then
	echo "Computing mean across all training vectors..."
	$train_cmd $work_dir/log/compute_mean.log \
  	 ivector-mean scp:$work_dir/train_xvector.scp \
  	 $work_dir/train_mean.vec || exit 1;
fi

if [ $stage -le 3 ]; then
	echo "Reducing dimensionality with LDA (to $lda_dim)..."
	$train_cmd $work_dir/log/lda.log \
	  ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
	  "ark:ivector-subtract-global-mean scp:$work_dir/train_xvector.scp ark:- |" \
	  ark:$work_dir/train_utt2spk $work_dir/train_transform.mat || exit 1;
fi

if [ $stage -le 4 ]; then
	echo "Training PLDA..."
	$train_cmd $work_dir/log/plda.log \
	  ivector-compute-plda ark:$work_dir/train_spk2utt \
	  "ark:ivector-subtract-global-mean scp:$work_dir/train_xvector.scp ark:- | transform-vec $work_dir/train_transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
	  $work_dir/plda || exit 1;
fi

if [ $stage -le 5 ]; then
	echo "Scoring..."
	$train_cmd $work_dir/log/scoring.log \
	    ivector-plda-scoring --normalize-length=true \
	    "ivector-copy-plda --smoothing=0.0 $work_dir/plda - |" \
	    "ark:ivector-subtract-global-mean $work_dir/train_mean.vec scp:$work_dir/train_xvector.scp ark:- | transform-vec $work_dir/train_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	    "ark:ivector-subtract-global-mean $work_dir/train_mean.vec scp:$work_dir/test_xvector.scp ark:- | transform-vec $work_dir/train_transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
	    "cat '$work_dir/trials' | cut -d\  --fields=1,2 |" $work_dir/scores || exit 1;
fi

if [ $stage -le 6 ]; then
	echo "Calculating EER, DET, accuracy and F1..."
	mkdir -p $output_dir/$variant
	python lda-plda-emotion/calculate_det_accuracy_and_f1.py --variant $variant --trials-file $work_dir/trials --score-file $work_dir/scores -o $output_dir/$variant | tee -a $output_dir/$variant/results.txt
fi
