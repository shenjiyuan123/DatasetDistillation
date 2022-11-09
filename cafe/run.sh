##### reproduce dc results#########
python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 10


########cross arch######
python all_ipc50.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --eval_mode M