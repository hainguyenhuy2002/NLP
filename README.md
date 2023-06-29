### Train Bartpho
```
cd BARTpho
```  
```
python train.py --train_data path/to/train/data --checkpoint_path path/to/folder/checkpoint
```   
### Generate candidates in BARTpho
```
cd BARTpho
```
```
python generate_candidates.py   
--train_data path/to/train/data   
--dev_data path/to/dev/data  
--test_data path/to/test/data   
--checkpoint_path path/to/model/checkpoint  
--save_folder path/to/candidates/folder
```
### Train SimCLS  
```
python main.py --cuda --gpuid 0 -l --val_step 500 --batch_size 2 --num_val 5000 --candidates_path path/to/candidates/folder
```
Your checkpoint will be saved as: **./SimCLS/checkpoints/.../scorer.pth**
### Eval SimCLS 
```
python main.py --cuda --gpuid 0 -e --model_pt ./SimCLS/checkpoints/.../scorer.pth --candidates_path path/to/candidates/folder
```
### Train Pointer Genrator Model
```
cd pointer-generator
```
You should prepare dataset and use **./make_datafiles.py** to generate data and vocabulary files. All your data then will be saved at **./dataset**.
Before training, you can adjust your desired hyperparameters and required data paths in **./utils/config.py**. There are options for training with rich features. To train the model, run the following command:
```
python train.py
```
To train with warming up, run:
```
python train.py -m path/to/model/checkpoint
```
Your checkpoints and logs will be saved as **./dataset/log/decode_model_50000_123456789/**.
### Eval Pointer Generator Model
```
cd pointer-generator
```
```
python test.py path/to/model/checkpoint
```
Your result will be saved as **./dataset/log/train_123456789/**.
### Infer summarization of a text
```
cd pointer-generator
python infer.py path/to/model/checkpoint "Your text that you want to summarize"
```
