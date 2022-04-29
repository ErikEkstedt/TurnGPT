train="python turngpt/train.py --gpus -1 --batch_size 2 --accumulate_grad_batches 10"

# Run on a small gpu so could not have large batch size thus the accumulate_grad_batches

# datasets:
# default is to use all
# "curiosity_dialogs",
# "daily_dialog",
# "multi_woz_v22",
# "meta_woz",
# "taskmaster1",
# "taskmaster2",
# "taskmaster3",
# Example:
# $train --datasets daily_dialog meta_woz

$train
