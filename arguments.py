import argparse

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',default='model/')
    parser.add_argument('--cached_save',default='cache/')
    parser.add_argument('--max_length',default=50000,type=int)
    parser.add_argument('--gpu_id',default=0,type=int)
    parser.add_argument('--batch_size',default=2,type=int)
    parser.add_argument('--epochs',default=20,type=int)
    parser.add_argument('--save_steps',default=500,type=int)
    parser.add_argument('--eval_steps',default=500,type=int)
    parser.add_argument('--logging_steps',default=500,type=int)
    parser.add_argument('--warmup_steps',default=500,type=int)
    parser.add_argument('--weight_decay',default=0.005,type=float)
    parser.add_argument('--lr',default=2e-3,type=float)
    parser.add_argument('--use_cache',action='store_true')
    parser.add_argument('--data_save',default='data/')
    parser.add_argument('--data_sounds',default="data/sound/")
    parser.add_argument('--data_transcripts',default="data/transcript/")
    parser.add_argument('--no_cuda',default=False,type=bool)
    args = parser.parse_args()
    return args

