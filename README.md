## Filtered Noise Shaping for Time Domain Room Impulse Response Estimation From Reverberant Speech

Implementation of the paper [Filtered Noise Shaping for Time Domain Room Impulse Response Estimation From Reverberant Speech](https://arxiv.org/abs/2107.07503).

This is a model that does blind estimation of room impulse response from reverberant speech.

### Setup
In the environment of your choice,
```
pip install -e .
pip install -r requirements.txt 
```

### Dataset
Integrate your dataset by modifying the code `fins/dataset/process_data.py` and check it by 
```
python fins/data/process_data.py
```
Example is given with a subset(fold01) of the BIRD dataset and DAPS speech dataset.

### Train
```
python -m fins.main
```

### Notes
* This is not the author's implementation, so little details, such as processing data, will be different. 