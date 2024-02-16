cd ../src/

python SSL_main.py --max_epochs 500 --experiment_name CIFAR10_SimCLR_MidTemp_$1 --run_index $1 --LossTemperature 0.5

cd ../scripts
./git_log_update.sh InfoNCE Mid temperature run
