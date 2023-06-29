Train Bartpho

```
cd BARTpho
```  

```
python train.py --train_data <train file path> --checkpoint_path <checkpoint folder>
```   

Generate candidates in BARTpho  

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
