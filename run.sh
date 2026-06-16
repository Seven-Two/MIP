python train.py --dataset Roman --method ewc --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method twp --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method ergnn --backbone SGC  --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method ssrmergnn --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method ssm --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method cat --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets
python train.py --dataset Roman --method puma --backbone SGC --gpu 0 --ILmode classIL --inter-task-edges False --minibatch False --ori_data_path /mnt/datasets