# **Image Colorization**

This repository provides a flexible framework for training various model combinations for image colorization using the **COCO2017 dataset**. It supports multiple generator and discriminator configurations, allowing users to explore and optimize their models with ease.

---

## **Supported Model Combinations**

- **UNetGenerator + PatchGANDiscriminator**  
  A classic architecture with a U-Net generator and a PatchGAN discriminator for generating realistic outputs.

- **ConvXNet (as backbone) + UNet + PatchGANDiscriminator**  
  Incorporates ConvXNet as the backbone for the U-Net generator to enhance feature extraction and learning capabilities.

- **ConvXNet (as backbone) + UNet + Vanilla Transformer + PatchGANDiscriminator**  
  Combines ConvXNet and U-Net with a Vanilla Transformer to capture both local and global dependencies for improved colorization results.

---

## **Configuration File**

The training process is fully configurable using a JSON configuration file. Below is an example:

```json
{
    "device": "cuda",
    "glb_min": 10,
    "initialize_weights": true,
    
    "gen_lr": 0.0002,
    "dic_lr": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    
    "train_dir": "/path/to/train/data",
    "val_dir": "/path/to/validation/data",
    "batch_size": 16,
    "num_workers": 2,
    
    "epochs": 5,
    "lambda_l1": 100,
    "show_interval": 100,

    "generator_path": "checkpoints/gen/best_gen.pth",
    "discriminator_path": "checkpoints/disc/best_disc.pth"
}
```

### **Key Parameters**
- **device**: Device to use for training (`"cuda"` or `"cpu"`).
- **glb_min**: Initial global minimum loss for tracking the best model.
- **initialize_weights**: Whether to initialize model weights.
- **gen_lr** & **dic_lr**: Learning rates for the generator and discriminator.
- **beta1**, **beta2**: Adam optimizer hyperparameters.
- **train_dir** & **val_dir**: Paths to training and validation datasets.
- **batch_size**: Number of samples per batch.
- **num_workers**: Number of data loading workers.
- **epochs**: Number of training epochs.
- **lambda_l1**: Weight for the L1 loss in the generator.
- **show_interval**: Interval (in batches) to display examples and save checkpoints.
- **generator_path** & **discriminator_path**: Paths for saving the best models.

---

## **Directory Structure**

The project follows a modular directory structure for better organization:

```
data/
    ├── data_preprocessing/
models/
    ├── model_definitions/
scripts/
    ├── eval.py
    ├── train.py
utils/
    ├── model_utils.py
    ├── train_utils.py
    ├── utils.py
```

### **Explanation**
- **data/**: Contains dataset and preprocessing scripts.
- **models/**: Stores model definitions and architecture files.
- **scripts/**: Includes training and evaluation scripts.
- **utils/**: Contains utility functions for training, model handling, and other auxiliary tasks.

---

## **Getting Started**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/image-colorization.git
   cd image-colorization
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   - Download the COCO2017 dataset and place the files in the specified `train_dir` and `val_dir`.

4. **Configure the Training**
   - Edit the JSON configuration file to match your system setup and requirements.

5. **Train the Model**
   ```python
   from scripts.train import ImageColorizationTrainer
   
   trainer = ImageColorizationTrainer("path/to/config.json")
   trainer.train() 

   trained_gen = trainer.generator
   trained_disc = trainer.discriminator
   ```

6. **Evaluate the Model**
   ```python
   from scripts.eval import eval_model
   eval_model(val_config, 'path/to/generator_epoch4_batch29571.pth')
   ```

---

## **Contributing**

Contributions are welcome! Please open an issue or submit a pull request for any bugs, features, or improvements.

---

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## **Acknowledgements**

This project uses the COCO2017 dataset. Special thanks to the contributors of this dataset for their valuable work.
