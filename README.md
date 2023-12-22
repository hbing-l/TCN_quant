# TCN Daily Frequency Quantity Price Factor

## Implementation of Changjiang Securities research report

use [T-63, T-1] to predict [T+1, T+3]

- TCN参数 [10, 10, 10, 20, 20] 随机选取200支股票用于训练 t+3 close - t+1 open  0.038
- TCN参数 [7, 7, 7, 7, 7] 随机选取200支股票用于训练 t+3 close - t+1 open  0.035
- TCN参数 [7, 7, 7, 7, 7] 随机选取200支股票用于训练 t+3 close - t+1 close  0.053
- TCN参数 [7, 7, 7, 7, 7] 随机选取200支股票用于训练 t+3 open - t+1 open  0.052
- TCN参数 [10, 10, 20, 20, 10] 随机选取200支股票用于训练 t+20 open - t+1 open lr=0.001  0.114 
- TCN参数 [10, 10, 20, 20, 10] 随机选取200支股票用于训练 t+20 open - t+1 open  lr=0.00095  0.113
- TCN参数 [10, 10, 20, 20, 10] 随机选取200支股票用于训练 t+20 open - t+1 open lr=0.001 5train+1valid+1test滚动训练  0.119 (val 0.161)
- TCN参数 [10, 10, 20, 20, 10] 随机选取200支股票用于训练 t+20 open - t+1 open lr=0.001 训练集从2017年开始依次增加1年滚动训练  0.105







