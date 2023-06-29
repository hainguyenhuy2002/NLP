### Train Bartpho
```
cd BARTpho
```  
```
python train.py --train_data <train file path> --checkpoint_path <checkpoint folder>
```   
### Generate candidates in BARTpho
```
cd BARTpho
```
```
python generate_candidates.py   
--train_data <train file path>   
--dev_data <dev file path>  
--test_data <test file path>   
--checkpoint_data <checkpoint BARTpho path>   
--save_folder <save folder>
```
### Train SimCLS  
**Note:** SimCLS will generate checkpoint with path: `SimCLS/checkpoints/.../scorer.pth`   
```
python main.py --cuda --gpuid 0 -l --val_step 500 --batch_size 2 --num_val 5000 --candidates_path <candidate folder>
```
### Eval SimCLS 
```
python main.py --cuda --gpuid 0 -e --model_pt <model path> --candidates_path <candidate folder>
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
