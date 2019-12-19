cd "iemocap" || exit
mkdir "data/kaldi/Text"
mkdir "data/kaldi/Speech"
mkdir "data/kaldi/Multimodal"

mv "data/kaldi/modified/MultimodalEmbeddings_xvector.1.ark" "data/kaldi/Multimodal/xvector.1.ark"
mv "data/kaldi/modified/MultimodalEmbeddings_xvector.2.ark" "data/kaldi/Multimodal/xvector.2.ark"
mv "data/kaldi/modified/MultimodalEmbeddings_xvector.3.ark" "data/kaldi/Multimodal/xvector.3.ark"
mv "data/kaldi/modified/MultimodalEmbeddings_xvector.4.ark" "data/kaldi/Multimodal/xvector.4.ark"
mv "data/kaldi/modified/MultimodalEmbeddings_xvector.5.ark" "data/kaldi/Multimodal/xvector.5.ark"

mv "data/kaldi/modified/SpeechEmbeddings_xvector.1.ark" "data/kaldi/Speech/xvector.1.ark"
mv "data/kaldi/modified/SpeechEmbeddings_xvector.2.ark" "data/kaldi/Speech/xvector.2.ark"
mv "data/kaldi/modified/SpeechEmbeddings_xvector.3.ark" "data/kaldi/Speech/xvector.3.ark"
mv "data/kaldi/modified/SpeechEmbeddings_xvector.4.ark" "data/kaldi/Speech/xvector.4.ark"
mv "data/kaldi/modified/SpeechEmbeddings_xvector.5.ark" "data/kaldi/Speech/xvector.5.ark"

mv "data/kaldi/modified/TextEmbeddings_xvector.1.ark" "data/kaldi/Text/xvector.1.ark"
mv "data/kaldi/modified/TextEmbeddings_xvector.2.ark" "data/kaldi/Text/xvector.2.ark"
mv "data/kaldi/modified/TextEmbeddings_xvector.3.ark" "data/kaldi/Text/xvector.3.ark"
mv "data/kaldi/modified/TextEmbeddings_xvector.4.ark" "data/kaldi/Text/xvector.4.ark"
mv "data/kaldi/modified/TextEmbeddings_xvector.5.ark" "data/kaldi/Text/xvector.5.ark"

cd "../"