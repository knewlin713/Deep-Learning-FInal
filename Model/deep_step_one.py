import shutil
import os
import sys
import ntpath
from shutil import copyfile
from distutils.dir_util import copy_tree
from synthesizer_preprocess_audio_tf import run_custom as syn_prep_audio
from synthesizer_preprocess_embeds_tf import run_custom as syn_prep_embeds
from synthesizeTrain_deeptalk_step_step1_tensorr_train_tf import run_custom as syn_train

def main(argv):
    input_path = argv[0]
    speaker_name = ntpath.basename(input_path)
    data_dir = 'Data'
    pretrained_model_name = 'model_GST'
    finetuned_model_name = pretrained_model_name + '_ft_' + speaker_name.lower()
    pretrained_encoder_model_path = os.path.join('trained_models', speaker_name, 'Encoder', pretrained_model_name + '.pt')
    finetuned_syn_model_dir = os.path.join('trained_models', speaker_name, 'Synthesizer', 'logs-' + pretrained_model_name + '_ft')
    syn_files_dir = os.path.join(data_dir, 'SV2TTS', 'synthesizer_' + finetuned_model_name)

    if not os.path.exists(syn_files_dir):
        os.makedirs(syn_files_dir)

    print("---------------------------------------------------------")
    print("Step1: Initialize the trained_model directory for the finetuning process using the generic pre-trained model in the trained_models directory ")

    generic_model_path = os.path.join('trained_models', 'Generic')
    finetuned_model_path = os.path.join('trained_models', speaker_name)

    if not os.path.isdir(finetuned_model_path):
        copy_tree(generic_model_path, finetuned_model_path)
        os.rename(os.path.join(finetuned_model_path, 'Synthesizer', 'logs-model_GST'), os.path.join(finetuned_model_path, 'Synthesizer', 'logs-model_GST_ft'))

    print("Step 1 Complete")
    print("---------------------------------------------------------")
    print('Step 2: Preprocess audio for training synthesizer')
    syn_prep_audio(data_dir, syn_files_dir, n_processes=8, skip_existing=True)
    print("Step 2 Complete")
    print("---------------------------------------------------------")
    print("Step 3: Get speaker embeddings from pre-processed audio for training synthesizer")
    syn_prep_embeds(syn_files_dir, pretrained_encoder_model_path, pretrained_model_name, n_processes=8, gpu_id='0') 
    print("Step 3 Complete")
    print("Run train_DeepTalk_step2.py now...")
    print("---------------------------------------------------------")
    print("Step 4: Finetune Synthesizer")
    syn_train(finetuned_model_name, pretrained_model_name, syn_files_dir, finetuned_syn_model_dir, gpu_id='0')
    print("Step 4 Complete")
    print("---------------------------------------------------------")
    print("----------------------------------------------------------")
    print('Run the following command to continue fine-tuning the model on the pre-processed data:')
    print('python train_DeepTalk_step2.py ' + input_path)
    print('----------------------------------------------------')

if __name__ == "__main__":
    main(sys.argv[1:])