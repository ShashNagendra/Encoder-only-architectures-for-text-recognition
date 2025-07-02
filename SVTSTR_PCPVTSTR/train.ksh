CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pcpvt_small_v0 --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pcpvt_small_v0

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pcpvt_base_v0 --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pcpvt_base_v0

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=pcpvt_large_v0 --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=pcpvt_large_v0

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=alt_gvt_small --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=alt_gvt_small

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=alt_gvt_base --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=alt_gvt_base

CUDA_VISIBLE_DEVICES=0 python3 train.py --train_data data_lmdb_release/training --valid_data data_lmdb_release/validation --select_data MJ-ST --batch_ratio 0.5-0.5 --Transformation None --FeatureExtraction None --SequenceModeling None --Prediction None --Transformer --TransformerModel=alt_gvt_large --imgH 224 --imgW 224 --manualSeed=123  --sensitive --batch_size=192 --exp_name=alt_gvt_large

