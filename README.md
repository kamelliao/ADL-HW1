# Homework 1 ADL NTU 111 Spring
## Task description

## Environment
### option 1: conda
```shell
# This builds a conda environment named 'adl-hw1'
make 
```

### option 2: pip
```bash
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Task 1: Intent detection
### Training
```shell
python3 ./src/train_intent.py --model_id=demo --model=CNN
```

### Testing
```shell
python3 ./src/test_intent.py --test_file="./data/intent/test.json" --pred_file="./ckpt/intent/demo/pred.csv" --model_file="./ckpt/intent/demo/model.pt" --model=CNN
```

## Task 2: Sequence tagging
### Training
```shell
python3 ./src/train_slot.py --model_id=demo 
```

### Testing
```shell
python3 ./src/test_slot.py --test_file="./data/slot/test.json" --pred_file="./ckpt/slot/demo/pred.csv" --model_file="./ckpt/slot/demo/model.pt"
```