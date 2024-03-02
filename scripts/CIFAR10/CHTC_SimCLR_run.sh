cd ../src/

python SSL_main.py --max_epochs 1000 --experiment_name CIFAR10_SimCLR_$1 --LossTemperature 0.5

cd ../scripts
# ./git_log_update.sh SimpleTest
