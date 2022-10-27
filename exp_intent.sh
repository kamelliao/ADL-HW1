for embed_type in 'last' 'sum' 'mean' 'learnt'
do
    for model in 'RNN_RELU' 'RNN_TANH' 'LSTM' 'GRU'
    do
        python3 ./src/train_intent.py --device=cuda --model_id=report --model="$model" --embed_type="$embed_type"
    done
done

for model in 'RNN_RELU 2' 'RNN_TANH 10' 'LSTM 2' 'GRU 10'
do
    set -- $i
    python3 ./src/train_intent.py --device=cuda --model_id=report --model="$1" --clip_grad_norm="$2"
done
