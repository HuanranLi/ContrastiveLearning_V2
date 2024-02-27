cd ../src/

python SSL_main.py --run_name lr10_MidTemp_SGD --max_epochs 15 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --experiment_name SymNSQ_Local_Exp --learning_rate 10
# python SSL_main.py --run_name InfoNCE_Bench --max_epochs 15 --accelerator mps --LossTemperature 0.5 --experiment_name SymNSQ_Local_Exp


cd ../scripts
