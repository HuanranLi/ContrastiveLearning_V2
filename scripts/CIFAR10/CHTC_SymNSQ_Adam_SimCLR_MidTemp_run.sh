cd ../src/

python SSL_main.py --max_epochs 1000 --LossTemperature 0.5 --criterion SymNSQ --experiment_name SymNSQ_ITER1000_Adam --optimizer Adam --learning_rate 1e-3 --run_name SymNSQ_ITER1000_Adam

cd ../scripts
