cd ../src/

python SSL_main.py --criterion SymNSQ --learning_rate 6e-2 --projection_head TransFusion-after_FNN --num_TF_layers 5 --experiment_name TFSymNSQ_3K --run_name TFSymNSQ_3K --max_epochs 3000 --LossTemperature 0.5

cd ../scripts
