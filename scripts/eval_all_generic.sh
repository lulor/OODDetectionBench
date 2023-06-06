#!/bin/bash 

dataset="$1"
src_domain="$2"
tgt_domain="$3"

echo "Running eval on $dataset with source $src_domain and target $tgt_domain"

echo "ReSeND"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resend --model resend --only_eval --evaluator resend --checkpoint_path pretrained_models/resend.pth

echo "Res101 CE KNN OOD"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator knn_ood 

echo "Res101 CE Mahalanobis"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator mahalanobis

echo "Res101 CE KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator knn_distance

echo "Res101 CE prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model CE --only_eval --evaluator prototypes_distance

for model in simclr supclr CSI supCSI
do
    echo "Res101 $model KNN"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator knn_distance --disable_contrastive_head

    echo "Res101 $model prototypes"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator prototypes_distance --disable_contrastive_head

    echo "Res101 $model KNN (contr. head)"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator knn_distance

    echo "Res101 $model prototypes (contr. head)"
    python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnet101 --model "$model" --checkpoint_path pretrained_models/"$model"_r101.pth --only_eval --evaluator prototypes_distance
done

echo "ViT-B CE KNN OOD"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator knn_ood 

echo "ViT-B CE Mahalanobis"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator mahalanobis

echo "ViT-B CE KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator knn_distance

echo "ViT-B CE prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE --only_eval --evaluator prototypes_distance

echo "ViT-B DINO KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator knn_distance --disable_contrastive_head

echo "ViT-B DINO prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator prototypes_distance --disable_contrastive_head

echo "ViT-B DINO KNN (contr. head)"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator knn_distance

echo "ViT-B DINO prototypes (contr. head)"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINO --checkpoint_path pretrained_models/DINO_vitb.pth --only_eval --evaluator prototypes_distance

echo "Vit-L CE ImageNet22k KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE-IM22k --only_eval --evaluator knn_distance

echo "Vit-L CE ImageNet22k prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model CE-IM22k --only_eval --evaluator prototypes_distance

echo "BiT resnetv2_101x3 KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnetv2_101x3 --model BiT --only_eval --evaluator knn_distance

echo "BiT resnetv2_101x3 prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network resnetv2_101x3 --model BiT --only_eval --evaluator prototypes_distance

echo "ViT-L CLIP MCM"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model clip --only_eval --evaluator MCM

echo "Vit-L DINOv2 KNN"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINOv2 --only_eval --evaluator knn_distance

echo "Vit-L DINOv2 prototypes"
python train.py --dataset "$dataset" --source "$src_domain" --target "$tgt_domain" --network vit --model DINOv2 --only_eval --evaluator prototypes_distance

