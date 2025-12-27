### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
cd src

###### Grocery_and_Gourmet_Food

### RPG
python main.py `
  --model_name RPG `
  --path ../data/ `  
  --dataset Grocery_and_Gourmet_Food `
  --gpu 0 `      
  --n_embd 64 `
  --n_inner 256 `    
  --n_layer 3 `    
  --n_head 4 `     
  --n_codebook 32 `
  --codebook_size 256 `
  --temperature 0.05 `
  --history_max 50 `
  --lr 0.001 `     
  --batch_size 1024 `
  --epoch 50 `
  --early_stop 20 `
  --num_workers 0

### SASRec
python main.py `
  --model_name SASRec `
  --path ../data/ `
  --dataset Grocery_and_Gourmet_Food `
  --gpu 0 `
  --history_max 50 `
  --batch_size 1024 `
  --early_stop 50 `
  --epoch 50 `
  --test_all 0 `
  --num_workers 0

### NARM
python main.py `
  --model_name NARM `
  --path ../data/ `
  --dataset Grocery_and_Gourmet_Food `
  --gpu 0 `
  --history_max 50 `
  --batch_size 1024 `
  --early_stop 50 `
  --epoch 50 `
  --test_all 0 `
  --num_workers 0

###### MovieLens_1M

### RPG
python main.py --model_name RPG `
  --path ../data/ `
  --dataset MovieLens_1M/ML_1MTOPK `
  --gpu 0 `
  --n_embd 64 `
  --n_inner 256 ` 
  --n_layer 3 `
  --n_head 4 `
  --n_codebook 32 `
  --codebook_size 256 `
  --temperature 0.05 `
  --history_max 50 `
  --lr 0.001 `
  --batch_size 1024 `
  --epoch 50 `
  --early_stop 20 `
  --num_workers 0

### SASRec
python main.py `
  --model_name SASRec `
  --path ../data/ `
  --dataset MovieLens_1M/ML_1MTOPK `
  --gpu 0 `
  --history_max 50 `
  --batch_size 1024 `
  --early_stop 50 `
  --epoch 50 `
  --test_all 0 `
  --num_workers 0

### NARM
python main.py `
  --model_name NARM `
  --path ../data/ `
  --dataset MovieLens_1M/ML_1MTOPK `
  --gpu 0 `
  --history_max 50 `
  --batch_size 1024 `
  --early_stop 50 `
  --epoch 50 `
  --test_all 0 `
  --num_workers 0
```


模型实现位于：`src/models/sequential/RPG.py`


### 主要参数说明

- `--n_embd`: GPT2的embedding维度（默认448）
- `--n_inner`: 前馈神经网络内部维度（默认1024）
- `--n_layer`: Transformer层数（默认2）
- `--n_head`: 注意力头数（默认4）
- `--n_codebook`: 语义ID的codebook数量（默认32）
- `--codebook_size`: 每个codebook的大小（默认256）
- `--temperature`: 温度参数，用于softmax（默认0.07）
- `--num_beams`: 图传播的beam数量（默认50）
- `--n_edges`: 图中每个节点的边数（默认50）
- `--propagation_steps`: 图传播步数（默认3）




