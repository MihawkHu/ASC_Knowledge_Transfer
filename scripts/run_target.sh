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
#python train_target.py --trans_way ab --model resnet --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 1 --latent_layers 'activation_9' 'activation_10' 'activation_21' 'activation_22' 'activation_33' --source_model ../ASC_Adaptation/exp_2020_resnet_baseline_source//model-62-0.7909.hdf5 --device ${device}
#python train_target.py --trans_way ab --model fcnn --lmd 0.1 --soft_ratio 0.9 --temperature 1.0 --alpha 0.0 --num_epochs_pretrain 1 --latent_layers 'activation_2' 'activation_4' 'activation_6' 'activation_8' --source_model ../ASC_Adaptation/exp_2020_fcnn_baseline_source//model-62-0.7939.hdf5 --device ${device}




done
