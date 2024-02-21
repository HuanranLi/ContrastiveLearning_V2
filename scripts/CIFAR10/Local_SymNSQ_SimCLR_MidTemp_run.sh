cd ../src/

python SSL_main.py --max_epochs 100 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --experiment_name SymNSQ_ITER100 #--learning_rate 1e-4

cd ../scripts
