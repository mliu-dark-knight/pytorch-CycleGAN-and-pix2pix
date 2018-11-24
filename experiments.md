## experiment_name
Without lambda_identities 



## c8r4_paints
share_two, 4 style transfer tasks, fix lambda identities
```
python train.py --context_size 8 --rank 4 --model cycle_gan  --task cezanne2photo,vangogh2photo,monet2photo,ukiyoe2photo --name c8r4_paints --lambda_identities 0.1  --batch_size 2 
python test.py --context_size 8 --rank 4 --model cycle_gan --task cezanne2photo,vangogh2photo,monet2photo,ukiyoe2photo --name c8r4_paints --lambda_identities 0.1 --epoch 20
```


## monet2photo_cg
baseline, workon branch master, task monet_photo
```
git checkout master
python train.py --dataroot ./datasets/monet2photo --name monet2photo_cg --model cycle_gan --batch_size 2 --lambda_identity 0.1
```


## c8r4_all
share_two, all tasks. Ongoing experiments on Tingfung's AWS.
```
python train.py --context_size 8 --rank 4 --model cycle_gan --name c8r4_all --lambda_identities 0.1  --batch_size 2 --save_epoch_freq 5
```
