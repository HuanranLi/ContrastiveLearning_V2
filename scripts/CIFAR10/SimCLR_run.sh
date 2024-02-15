cd ../src/

python main.py --max_epochs 100 --experiment_name CIFAR10_$1 --run_index $1

cd ../scripts
./git_log_update.sh SimpleTest
