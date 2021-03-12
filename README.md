# CycleGAN-pytorch

# Prepare data

path: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/


Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. monet2photo
    |   |   ├── train              # Training
    |   |   |   ├── A              # Contains domain A images (i.e. monet)
    |   |   |   └── B              # Contains domain B images (i.e. photo)
    |   |   └── test               # Testing
    |   |   |   ├── A              # Contains domain A images (i.e. monet)
    |   |   |   └── B              # Contains domain B images (i.e. photo)

# Train

python train.py --data_dir=data/monet2photo --output_dir=output/model/monet2photo


# Test
python run_test.py --image_path=images/test1.jpg --model_path=output/model/monet2photo/g_BA.pt