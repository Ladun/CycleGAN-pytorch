# CycleGAN-pytorch

# Prepare data

path: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/


Alternatively you can build your own dataset by setting up the following directory structure:

    .
    ├── datasets                   
    |   ├── <dataset_name>         # i.e. monet2photo
    |   |   ├── trainA             # Contains domain A train images (i.e. monet)
    |   |   ├── trainB             # Contains domain B train images (i.e. photo)
    |   |   ├── testA              # Contains domain A test images (i.e. monet)
    |   |   └── testB              # Contains domain B test images (i.e. photo)

# Train

```
python train.py --data_dir=data/monet2photo --output_dir=output/model/monet2photo
```

# Test

## Test one file

```
python infer.py --image_path=path/to/test1.jpg --model_path=path/to/g_BA.pt [--output_path=path/to/output_image.jpg]
```

## Test directory

```
python infer.py --image_path=path/to/directory --model_path=path/to/g_BA.pt [--output_path=path/to/directory]
```
