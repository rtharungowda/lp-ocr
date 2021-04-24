
# Training a character recognition model

+ This *readme* describes the training procedure involved in optical character recognition of license plate characters
+ Edit and execute files in the following order and change paths where ever necessary

    1. `dataset_characters.zip` - dataset to train on
    2. `utils.py` - contains all the necessary plot, load and save model helper functions.
    3. `config.py` - contains all the necessary hyperparameters and paths to dataset and also the character to numeric mapping.
    4. `pre_csv.py` - creates a csv file for train and test, which contains the path to images along with its label.
    5. `dataloader.py` - implements augmentation and preproceesing on custom dataset and clubs images into batches using dataloader.
    6. `model.py` (used for experimentaiton) - contains pretrained and custom models which were used for experimentaiton. We finally decided to use ResNet34 pretrained on Imagenet dataset, which gave a 98.75 validation accuracy.
    7. `train.py` - train model

Execute a file

```shell
python3 <file_name>.py
```