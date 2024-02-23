cd ../src/

python SSL_main.py --run_name TF5_Info_Adam_ITER1000 --num_TF_layers 5 --experiment_name TF5_Info_Adam_ITER1000 --max_epochs 1000 --LossTemperature 0.5 --projection_head TransFusion --learning_rate 1e-3 --optimizer Adam

cd ../scripts
