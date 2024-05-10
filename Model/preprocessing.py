from synthesizer_tf.synthesize import run_synthesis_tf
from synthesizer_tf.hparams import hparams_tf
from utils_tf.argutils import print_args_tf
import argparse
import os

def run_custom_tf(in_dir, out_dir, model_dir):
    run_synthesis_tf(in_dir, out_dir, model_dir, hparams_tf)

def main():
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Creates ground-truth aligned (GTA) spectrograms from the vocoder.",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=str, help=\
        "Path to the directory containing your SV2TTS directory. If you specify both --in_dir and "
        "--out_dir, this argument won't be used.")
    parser.add_argument("--model_dir", type=str,
                        default="synthesizer_tf/saved_models/logs-pretrained/", help=\
        "Path to the pretrained model directory.")
    parser.add_argument("-i", "--in_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the mel spectrograms, the wavs and the "
        "embeds. Defaults to  <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-o", "--out_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the output vocoder directory that will contain the ground truth aligned mel "
        "spectrograms. Defaults to <datasets_root>/SV2TTS/vocoder/.")
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value "
                             "pairs")
    parser.add_argument("-gpuid", "--gpu_id", type=str, default='0', help= \
        "Select the GPU to run the code")
    args = parser.parse_args()
    print_args_tf(args, parser)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    modified_hp = hparams_tf.parse(args.hparams)

    if not hasattr(args, "in_dir"):
        args.in_dir = os.path.join(args.datasets_root, "SV2TTS", "synthesizer")
    if not hasattr(args, "out_dir"):
        args.out_dir = os.path.join(args.datasets_root, "SV2TTS", "vocoder")

    run_synthesis_tf(args.in_dir, args.out_dir, args.model_dir, modified_hp)

if __name__ == "__main__":
    main()