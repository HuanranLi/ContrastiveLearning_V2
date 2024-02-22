cd ../src/

python SSL_main.py --run_name TF1_SymNSQ_ITER1000 --num_TF_layers 1 --experiment_name TF1_SymNSQ_ITER1000 --max_epochs 1000 --LossTemperature 0.5 --criterion SymNSQ --projection_head TransFusion

cd ../scripts
