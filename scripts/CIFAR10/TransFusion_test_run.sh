cd ../src/

# python SSL_main.py --run_name 10_Layer_TF --num_TF_layers 10 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --projection_head TransFusion
# python SSL_main.py --run_name 5_Layer_TF_Info --num_TF_layers 5 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5 --projection_head TransFusion
# python SSL_main.py --run_name Info_bench --num_TF_layers 5 --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5

learning_rate=1e-5
projection_head=TransFusionV2
num_TF_layers=5

run_name=SymNSQ_${projection_head}_Layer${num_TF_layers}_LR${learning_rate}


python SSL_main.py --criterion SymNSQ --learning_rate $learning_rate --run_name $run_name --projection_head $projection_head --num_TF_layers $num_TF_layers --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 5 --accelerator mps --LossTemperature 0.5


cd ../scripts
