cd ../src/

# python SSL_main.py --run_name 10_Layer_TF --num_TF_layers 10 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --projection_head TransFusion
# python SSL_main.py --run_name 5_Layer_TF_Info --num_TF_layers 5 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5 --projection_head TransFusion
python SSL_main.py --run_name Info_bench --num_TF_layers 5 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5
cd ../scripts
