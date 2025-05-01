# Hidformer-Hierarchal-dual-tower-tranfomer

Hidformer: Hierarchical dual-tower transformer using multi-scale mergence for long-term time series forecasting applied to the stock domain

# How to Run

To run this model, start by cloning the repository.

Once you have done that you may run
'pip install -r requirements.txt'

This will install python packages needed to run this project.

Now you are ready to either train your own model or run a model that has already been trained.

## To Train

To train this model, you must go into the train.py file. in there, there is logic for training a model based on many constants set in the file.

The training data will be prepared based on these settings and with the 'tickers.csv' file in src/

In this file, you are able to list any stocks by their ticker and their entire market history will be added to the training data.

The data is automatically split in a ratio of 75% training 15% validation, and 15% testing.

The data is split in a way that the most recent 30% of any given stocks price history is not included in the training data.

## Inference

If you would like to do inference with this model, you may use the file basic_inference.py.

You must use command line argumetns that match the model's parameters

'python src/basic_inference.py --ticker "AAPL" --start_date "2020-02-24" --model_path "./model/128_128_4_2_128_0.2_256.pt" --target_column "Close" --lookback_window 128 --prediction_horizon 128 --feature_columns Open High Low Close Volume --token_length 32 --stride 16 --num_time_blocks 4 --num_freq_blocks 2 --d_model 128 --freq_k 256 --dropout 0.2 --merge_mode "linear" --merge_k 2'
