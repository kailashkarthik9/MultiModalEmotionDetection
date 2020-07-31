. ./cmd.sh
. ./path.sh

set -e

nj=placeholder
use_gpu=placeholder
model_path=placeholder
num_layers=placeholder
corpus_dir=placeholder
output_base_dir=placeholder

. ./utils/parse_options.sh

start_layer=6
min_chunk_size=25
max_chunk_size=10000

for layer in $(eval echo "{$start_layer..$num_layers}")
do
	echo "Extracting x-vectors from layer ${layer} of ${model_path}..."
	
	# first we configure the extraction (which layer) and then copy the source
	# model into a temporary working directory (to avoid corrupting our data)
	work_dir="temp"
	mkdir $work_dir
	echo "$min_chunk_size" > $work_dir/min_chunk_size
	echo "$max_chunk_size" > $work_dir/max_chunk_size
	echo "output-node name=output input=tdnn${layer}.affine" > $work_dir/extract.config
	cp $model_path $work_dir/final.raw

	out_dir="$output_base_dir/$layer"
	mkdir $out_dir
	sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj --use_gpu $use_gpu \
		$work_dir $corpus_dir $out_dir

	# clean up (so we hard fail if we run into any issues on the next iteration)
	rm -rf $work_dir
done
