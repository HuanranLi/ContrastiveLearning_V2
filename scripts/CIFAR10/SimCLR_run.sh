cd ../src/

python main.py --max_epochs 10 --experiment_name CIFAR10 --run_index $1

cd ../scripts
./git_log_update.sh SimpleTest
