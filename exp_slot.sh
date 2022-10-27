for model in 'RNN_RELU' 'RNN_TANH' 'LSTM' 'GRU' 'CNN-biLSTM'
do 
    python3 ./src/train_slot.py --device=cuda --model_id=report --dropout=0.7 --model="$model"
done