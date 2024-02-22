cd ../src/

python SSL_main.py --max_epochs 2 --accelerator mps --LossTemperature 0.5 --criterion SymNSQ --batch_size 32 --projection_head TransFusion

cd ../scripts
