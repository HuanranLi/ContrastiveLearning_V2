cd ../src/

python main.py --max_epochs 100 --experiment_name CIFAR10 --run_index $1

cd ../scripts
./gitupdate.sh SimpleTest
