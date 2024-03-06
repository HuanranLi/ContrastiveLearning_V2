cd ../src/

python SSL_main.py --criterion SymNSQ --learning_rate 6e-2 --projection_head TransFusion-after_FNN --num_TF_layers 5 --experiment_name TFSymNSQ_After_FNN_1K --run_name TFSymNSQ_After_FNN_1K --max_epochs 1000 --LossTemperature 0.5

cd ../scripts
