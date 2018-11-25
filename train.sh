#!/bin/sh

suffix=_baseline
for dataset in apple2orange cezanne2photo cityscapes facades horse2zebra maps monet2photo summer2winter_yosemite ukiyoe2photo vangogh2photo; do
    command="python train.py --model cycle_gan --dataroot ./datasets/$dataset --name $dataset$suffix --batch_size 5 --lambda_identity 0.1"
    echo $command
    $command
done
