from typing import Any, Iterator
import torch
import pandas as pd
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

class TCN_data(Dataset):
    def __init__(self, dict_data, timepoints, year, mode="train", T=63, H=20):
        super(TCN_data, self).__init__()
        self.fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
        # test_data = pd.read_csv(filename)
        # test_data = test_data.fillna(0.)
        self.data1 = dict_data
        # self.data1 = {f:test_data.pivot_table(index='datetime', columns='instrument', values=f) for f in self.fields} # 每一天 每一支股票的属性


        self.time_points = timepoints
        self.time_points.sort()
        
        
        train_set_point = len(self.time_points[self.time_points<str(year)+'-01-01'])
        valid_set_point = len(self.time_points[(self.time_points>=str(year)+'-01-01') & 
                                        (self.time_points<str(year+1)+'-01-01')])
        test_set_point = len(self.time_points[(self.time_points>=str(year+1)+'-01-01') & 
                                        (self.time_points<str(year+2)+'-01-01')])
        self.mode = mode
        # assert train_set_point + valid_set_point + test_set_point == len(self.time_points)
        if mode == 'train':
            self.data_range = list(range(T, train_set_point))
        elif mode == 'valid':
            self.data_range = list(range(train_set_point, train_set_point+valid_set_point))
        else:
            self.data_range = list(range(train_set_point+valid_set_point, train_set_point+valid_set_point+test_set_point-H))
            
        # start_point = len(self.time_points[self.time_points<str(year)+'-01-01'])
        # train_set_point = len(self.time_points[(self.time_points>=str(year)+'-01-01') &
        #                                        (self.time_points<str(year+5)+'-01-01')])
        # valid_set_point = len(self.time_points[(self.time_points>=str(year+5)+'-01-01') & 
        #                                 (self.time_points<str(year+6)+'-01-01')])
        # test_set_point = len(self.time_points[(self.time_points>=str(year+6)+'-01-01') & 
        #                                 (self.time_points<str(year+7)+'-01-01')])
        # self.mode = mode
        # # assert train_set_point + valid_set_point + test_set_point == len(self.time_points)
        # if mode == 'train':
        #     self.data_range = list(range(start_point+T, start_point+train_set_point))
        # elif mode == 'valid':
        #     self.data_range = list(range(start_point+train_set_point, start_point+train_set_point+valid_set_point))
        # else:
        #     self.data_range = list(range(start_point+train_set_point+valid_set_point, start_point+train_set_point+valid_set_point+test_set_point-H))
        # split_point = len(self.time_points[self.time_points<'2020-01-01'])
    def __len__(self):
        return len(self.data_range)
    def get_batch(self, t, T=63, H=20):
        # assert t>=T and t<len(self.time_points)-H
        tmp = []
        close_data = self.data1['close']
        open_data = self.data1['open']
        
        valid_columes_idx = close_data.loc[self.time_points[t-T]: self.time_points[t+H]].isnull().any()
        valid_columes = list(valid_columes_idx.loc[valid_columes_idx.values == False].index)
        
        # valid_columes_idx = [self.data1['close'][i].loc[self.time_points[t-T]: self.time_points[t+H]].isnull().any() for i in columns]
        # for i in range(len(columns)):
        #     if not valid_columes_idx[i]:
        #         valid_columes.append(columns[i])
        valid_close_data = close_data[valid_columes]
        valid_open_data = open_data[valid_columes]
        closing_price_at_start = valid_close_data.loc[self.time_points[t-T]]
        
        if self.mode == 'train':
            if valid_close_data.columns.shape[0] >= 200:
                selected_indices = np.random.choice(valid_close_data.columns, size=200, replace=False)
            else:
                selected_indices = valid_close_data.columns
        else:
            selected_indices = valid_close_data.columns
        for i, f in enumerate(self.fields):
            current_data = self.data1[f][valid_columes]
            # closing_price_at_start = self.data1['close'].loc[self.time_points[t-T]]
            if f in ['open', 'high', 'low', 'close', 'vwap']:
                _field = current_data.loc[self.time_points[t-T]: self.time_points[t-1]] / closing_price_at_start
                selected_field = _field[selected_indices]
                
            else:
                min_num = current_data.loc[self.time_points[t-T]: self.time_points[t-1]].min()
                max_num = current_data.loc[self.time_points[t-T]: self.time_points[t-1]].max()
                _field = (current_data.loc[self.time_points[t-T]: self.time_points[t-1]] - min_num) / (max_num - min_num)
                selected_field = _field[selected_indices]

            selected_field = selected_field.fillna(0.)
            tmp.append(selected_field.values)
            
        prediction = (valid_open_data.loc[self.time_points[t+H]] - valid_open_data.loc[self.time_points[t+1]]) / valid_open_data.loc[self.time_points[t+1]]
        selected_prediction = prediction[selected_indices]
        selected_prediction = selected_prediction.fillna(0.)
        percentile_ranks = selected_prediction.rank(pct=True)
        # prediction = prediction.fillna(0.)
        Y = percentile_ranks.values
        # Y = prediction.values
        
        X = np.transpose(np.stack(tmp), [2, 0, 1])

        return X, Y
    
    def __getitem__(self, index) -> Any:
        return self.get_batch(self.data_range[index])
    

def pearson_correlation(x, y):
    """
    Compute the Pearson correlation coefficient between
    two tensors while preserving gradient information.
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den

    # To avoid division by zero, in case of zero variance
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    return r
  
class CustomLoss(nn.Module):
    def __init__(self, lambda1, lambda2):
        super(CustomLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # self.n2 = n2

    def forward(self, outputs, target):
        
        """
        outputs 维度: T*64 T是股票个数，64 是因子个数
        targets: 维度 T
        """
        
        # Convert target to a float tensor in case it's not
        # target = target.float()

        # Ensure that target is a 2D row vector
        target = target.view(1, -1)

        # Calculate correlation for each factor with the target
        # Pearson correlation coefficient is used here
        corrs = [pearson_correlation(outputs[:, i], target).unsqueeze(0) for i in range(outputs.shape[1])]
        corrs = torch.cat(corrs)
        
        # Calculate the first term of the loss function
        term1 = -torch.mean(corrs)

        # Calculate the sum of correlations for the second term
        # sum_outputs =
        sum_output = torch.sum(outputs, 1)
        sum_corr = pearson_correlation(sum_output, target)
        # sum_corr = torch.sum(corrs)

        # Calculate the second term of the loss function
        term2 = -self.lambda1 * sum_corr

        # Calculate the squared correlations for the third term
        # corr_squared = corrs**2
        # pair_wise_corr = torch.cat([pearson_correlation(outputs[:, i], outputs[:, j]).unsqueeze(0) for i in range(outputs.shape[1]) for j in range(outputs.shape[1])]) ** 2
        

        # n_squared = outputs.shape[1] ** 2

        # Calculate the third term of the loss function
        term3 = self.lambda2 * torch.mean(torch.corrcoef(outputs.T) ** 2)

        # Sum all terms to get final loss
        loss_final = term1 + term2 + term3

        return loss_final

def corr_loss(yhat, y, lambda1=0.5, lambda2=1):
    corr = 0.
    for i in range(yhat.shape[1]):
        r = torch.concat((yhat[:, i].unsqueeze(1), y), dim=1).transpose(1, 0)
        corr += torch.corrcoef(r)[0, 1]
    corr1 = -corr / yhat.shape[1]
    # print(corr1)

    r = torch.sum(yhat, dim=1).unsqueeze(1)
    r = torch.concat((r, y), dim=1).transpose(1, 0)
    corr2 = -torch.corrcoef(r)[0, 1]
    # print(corr2)

    correlation_matrix = torch.corrcoef(yhat.transpose(1, 0))
    # corr3 = torch.mean(correlation_matrix**2) - torch.mean(torch.diagonal(correlation_matrix)**2)
    corr3 = torch.mean(correlation_matrix**2)

    # corr = 0.
    # for i in range(yhat.shape[1]):
    #     for j in range(yhat.shape[1]):
    #         r = torch.concat((yhat[:, i].unsqueeze(1), yhat[:, j].unsqueeze(1)), dim=1).transpose(1, 0)
    #         corr += torch.corrcoef(r)[0, 1] ** 2
    # corr3 = corr / (yhat.shape[1] ** 2)
    # print(corr3)
    
    total_loss = corr1 + lambda1 * corr2 + lambda2 * corr3
    return total_loss


def correlation_coefficient_loss(output, target):
    x = output - output.mean()
    y = target - target.mean()
    r_num = torch.sum(x * y)
    r_den = torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2))
    r = r_num / r_den
    return r
        
def custom_loss(output, target, lambda1, lambda2):
    factor_correlations = []
    total_corr_loss = 0.0
    for i in range(output.size(1)):
        corr_loss = correlation_coefficient_loss(output[:, i], target)
        total_corr_loss += corr_loss
        factor_correlations.append(output[:, i])

    # Average correlation loss for individual factors
    avg_corr_loss = total_corr_loss / output.size(1)
    # print(avg_corr_loss)

    # Composite factor correlation loss
    composite_factor = torch.mean(output, dim=1)
    composite_corr_loss = correlation_coefficient_loss(composite_factor, target)
    # print(composite_corr_loss)

    # Correlation among factors
    correlation_matrix = torch.corrcoef(torch.stack(factor_correlations))
    factor_correlation_loss = torch.mean(correlation_matrix**2) - torch.mean(torch.diagonal(correlation_matrix)**2)
    # factor_correlation_loss = torch.mean(correlation_matrix**2)
    # print(factor_correlation_loss)

    # Total loss
    total_loss = -avg_corr_loss - lambda1 * composite_corr_loss + lambda2 * factor_correlation_loss
    return total_loss
# custom_loss(yhat, y.squeeze(), 0.5, 1) 


if __name__ == '__main__':
    print("reading data ... ...")
    df_data = pd.read_csv("tcn_data.csv")
    df_data = df_data.fillna(0.)
    # df_data.to_pickle("tcn_data.pkl")
    # quit()
    time_points = df_data["datetime"].unique()
    # test_data = test_data.fillna(0.)  
    fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
    dict_data = {f:df_data.pivot_table(index='datetime', columns='instrument', values=f) for f in fields} # 每一天 每一支股票的属性

    print("loading train/valid/test set ,,, ,, ")
    train_data = TCN_data(dict_data, time_points, "train")
    valid_data = TCN_data(dict_data, time_points, "valid")
    test_data = TCN_data(dict_data, time_points, "test")
    print(test_data[0])
    print(len(train_data), len(valid_data), len(test_data))
    
    T = 10  # Let's say we have 10 stocks
    outputs = torch.randn(T, 64, requires_grad=True)
    target = torch.randn(T)
    custom_loss1 = corr_loss(outputs, target.unsqueeze(1))
    custom_loss2 = custom_loss(outputs, target, 0.5, 1)

    # print(f"Calculated loss: {loss.item()}")