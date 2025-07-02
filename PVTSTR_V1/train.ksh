CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pvt_tiny --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pvt_tiny

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pvt_small --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pvt_small

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pvt_medium --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pvt_medium

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pvt_large --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pvt_large

