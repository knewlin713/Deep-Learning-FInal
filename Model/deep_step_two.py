import os
import sys
import ntpath
from shutil import rmtree
from shutil import copyfile
from distutils.dir_util import copy_tree
from synthesizer_preprocess_audio_tf import run_custom as syn_prep_audio
from vocoder_preprocess_tf import run_custom as voc_prep
from vocoder_train_tf import run_custom as voc_train

def main(argv):

    input_path = argv[0]
    speaker_name = ntpath.basename(input_path)
    data_dir = os.path.abspath('Data')
    pretrained_model_name = 'model_GST'
    finetuned_model_name = pretrained_model_name + '_ft_' + speaker_name.lower()
    finetuned_syn_model_dir = os.path.join('trained_models', speaker_name, 'Synthesizer', 'logs-' + pretrained_model_name + '_ft')
    finetuned_voc_model_dir = os.path.join('trained_models', speaker_name, 'Vocoder')     
    syn_files_dir = os.path.join(data_dir, 'SV2TTS', 'synthesizer_' + finetuned_model_name)    
    voc_files_dir = os.path.join(data_dir, 'SV2TTS', 'vocoder_' + finetuned_model_name)

    if not os.path.exists(syn_files_dir):
        os.makedirs(syn_files_dir)
    if not os.path.exists(finetuned_syn_model_dir):
        os.makedirs(finetuned_syn_model_dir)
    if not os.path.exists(voc_files_dir):
        os.makedirs(voc_files_dir)
    if not os.path.exists(finetuned_voc_model_dir):
        os.makedirs(finetuned_voc_model_dir)

    print("---------------------------------------------------------")
    print("Step 5: Preprocess data for training the Vocoder")
    voc_prep(syn_files_dir, voc_files_dir, finetuned_syn_model_dir)
    print("Step 5 Complete")
    print("---------------------------------------------------------")
    print("Step 6: Finetune Vocoder")
    voc_train(pretrained_model_name + '_ft', syn_files_dir, voc_files_dir, finetuned_voc_model_dir)
    print("Step 6 Complete")
    print("---------------------------------------------------------")
    # Remove data from SV2TTS directory
    rmtree(syn_files_dir)
    rmtree(voc_files_dir)

    print("-----------------Finetuning Completed!!----------------------")


if __name__ == "__main__":
    main(sys.argv[1:])