**Train Bartpho**
```
cd BARTpho
```  
```
python train.py --train_data <train file path> --checkpoint_path <checkpoint folder>
```   
**Generate candidates in BARTpho**  
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
**Train SimCLS**   
***Note:*** SimCLS will generate checkpoint with path: `SimCLS/checkpoints/.../scorer.pth`   
Example config:
```
python main.py --cuda --gpuid 0 -l --val_step 500 --batch_size 2 --num_val 5000 --candidates_path <candidate folder>
```
**Eval CLS**   
Example config:
```
python main.py --cuda --gpuid 0 -e --model_pt <model path> --candidates_path <candidate folder>
```
