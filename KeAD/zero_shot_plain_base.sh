#!/bin/bash

# few_shots=(0 1 2 4)
few_shots=(4)

for few_num in "${!few_shots[@]}";do

    base_name=KeAD  # winclip_mvtec
    des_path=./data/text_description/datasets_des_info_ori_ori.json  # datasets_des_info_ori_gpt4o
    meta_path=./data/dataset_to_public.json
    #des_path=/data/datasets/pub/public/own_anomaly_detect/datasets_des_info.json
    #meta_path=/data/datasets/pub/public/own_anomaly_detect/metal_meta_d4.json
    surgery_type=vv_res
    dataset_name=metal_own
    data_path=/data/datasets/pub/public/own_anomaly_detect

    save_dir=./output/exps_${base_name}/${dataset_name}_vit_base_16_240_few_shot_${few_shots[few_num]}_${surgery_type}_ori_gpt4o/
    #save_dir=./output/exps_${base_name}/mvtecvit_huge_14_378_few_shot_${few_shots[few_num]}/

    CUDA_VISIBLE_DEVICES=2 python -u main/get_anomaly_map_base_ori.py --dataset ${dataset_name} \
    --save_path ${save_dir} --data_path ${data_path}\
    --des_path ${des_path} --meta_path ${meta_path} \
    --model ViT-B-16-plus-240 --pretrained laion400m_e32 --k_shot ${few_shots[few_num]} \
    --image_size 240 --patch_size 16 --feature_list 3 6 9 12 --dpam_layer 10 \
    --surgery_type ${surgery_type} --use_detailed
    wait
done
#--surgery_type vv \
    #--visualize --save_anomaly_map  --use_detailed  --dpam_layer 10

# --model ViT-H-14-378-quickgelu --pretrained dfn5b --k_shot ${few_shots[few_num]} --image_size 378 --patch_size 14 --feature_list 8 16 24 32 --dpam_layer 26 
#--model ViT-B-16-plus-240 --pretrained laion400m_e32 --k_shot ${few_shots[few_num]} --image_size 240 --patch_size 16 --feature_list 3 6 9 12 --dpam_layer 10 

