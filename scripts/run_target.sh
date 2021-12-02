#for device in b c s1 s2 s3 s4 s5 s6
for device in b
do


# tsl
#python train_target.py --trans_way tsl --model resnet --lmd 0.0 --soft_ratio 1.0 --temperature 1.0 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way tsl --model fcnn --lmd 0.0 --soft_ratio 1.0 --temperature 1.0 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# nle
#python train_target.py --trans_way nle --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --nle_path tools/nle.txt --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way nle --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --nle_path tools/nle.txt --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# fitnets
#python train_target.py --trans_way fitnets --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 2.0 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way fitnets --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 2.0 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# at
#python train_target.py --trans_way at --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1000 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way at --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1000 --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}

# ab
#python train_target.py --trans_way ab --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 30 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way ab --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 30 --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# vid
#python train_target.py --trans_way vid --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1.0 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way vid --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1.0 --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# fsp 
#python train_target.py --trans_way fsp --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 30 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way fsp --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 30  --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# cofd
#python train_target.py --trans_way cofd --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.5 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way cofd --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.5 --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# sp
#python train_target.py --trans_way sp --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 300 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way sp --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 300 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# cckd
#python train_target.py --trans_way cckd --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.02 --cckd_emb_size 128 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way cckd --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.02 --cckd_emb_size 128 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# pkt
#python train_target.py --trans_way pkt --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 30000 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way pkt --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 30000 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


# nst
#python train_target.py --trans_way nst --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 5.0 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way nst --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 5.0 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}

# nst
#python train_target.py --trans_way rkd --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 25.0 --beta 50.0 --latent_layer activation_33 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way rkd --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 25.0 --beta 50.0 --latent_layer activation_8 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}

# vbkt
#python train_target.py --trans_way vbkt --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1.0 --sigma 0.2 --latent_layer batch_normalization_35 --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
python train_target.py --trans_way vbkt --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 1.0 --sigma 0.2 --latent_layer batch_normalization_9 --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}


done
