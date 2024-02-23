cd ../src/

python SSL_main.py --optimizer Adam --run_name lr1e-4_MidTemp_Adam --max_epochs 15 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --experiment_name SymNSQ_Local_Exp --learning_rate 1e-4
# python SSL_main.py --run_name InfoNCE_Bench --max_epochs 15 --accelerator mps --LossTemperature 0.5 --experiment_name SymNSQ_Local_Exp


cd ../scripts
