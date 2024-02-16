cd ../src/

python SSL_main.py --max_epochs 500 --experiment_name CIFAR10_SimCLR_$1 --run_index $1

cd ../scripts
./git_log_update.sh SimpleTest
