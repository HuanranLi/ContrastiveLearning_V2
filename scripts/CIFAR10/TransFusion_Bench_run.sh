cd ../src/

python SSL_main.py --run_name SimCLR_baseline --experiment_name TransFusion_MidTemp_ITER30 --max_epochs 15 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ 

cd ../scripts
