import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from model import TCN
from utils import TCN_data, corr_loss
import logging
import time
from torch.utils.tensorboard import SummaryWriter
localtime = time.localtime(time.time())
time1 = time.strftime("%m%d_%H%M",time.localtime(time.time()))
writer = SummaryWriter('/home/hbliu/TCN_quant/runs/{}/'.format(time1))


def train(model, train_loader, epoch, optimizer, lambda1, lambda2):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.float().squeeze(0).cuda()
            target = target.float().squeeze(0).unsqueeze(1).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = corr_loss(output, target, lambda1, lambda2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # break
    return total_loss / len(train_loader)

def evaluate_on_validation_dataset(model, validation_dataloader):
    """
    Evaluate the model on the validation dataset to calculate mean IC and mean Rank IC.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    validation_dataloader (torch.utils.data.DataLoader): The DataLoader providing the validation dataset.

    Returns:
    Tuple[float, float]: The mean IC and mean Rank IC for the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode.
    ic_scores = []
    rank_ic_scores = []
    total_loss = 0
    with torch.no_grad():  # No gradient computation during evaluation.
        for features, targets in validation_dataloader:
        # for features, targets in tqdm(validation_dataloader, total = len(validation_dataloader)):
            if torch.cuda.is_available():
                features, targets = features.float().squeeze(0).cuda(), targets.float().squeeze(0).unsqueeze(1).cuda()
            outputs = model(features)
            loss = corr_loss(outputs, targets)
            ic, rank_ic = evaluate_ic_rankic(outputs, targets)
            ic_scores.append(ic)
            rank_ic_scores.append(rank_ic)
            total_loss += loss.item()
            # break
    # Compute the mean scores across all validation data batches
    mean_ic = sum(ic_scores) / len(ic_scores)
    mean_rank_ic = sum(rank_ic_scores) / len(rank_ic_scores)
    
    return mean_ic, mean_rank_ic, total_loss / len(validation_dataloader)

def evaluate_ic_rankic(outputs, target):
    """
    Evaluate the Information Coefficient (IC) and Rank Information Coefficient (Rank IC)
    between model outputs and actual labels.

    Parameters:
    outputs (Tensor): The model outputs, expected to be of shape (T, num_factors)
    target (Tensor): The actual labels, expected to be of shape (T,)

    Returns:
    Tuple[float, float]: A tuple containing the IC and Rank IC values.
    """
    # Ensure target is a float tensor
    target = target.float()

    # Ensure target is a 2D column vector
    target = target.view(-1, 1)

    # Calculate IC
    # Pearson correlation coefficient between the mean of the outputs and the target
    mean_outputs = torch.mean(outputs, dim=1)
    ic = torch.corrcoef(torch.stack((mean_outputs, target.squeeze())))[0, 1]

    # Calculate Rank IC
    # Spearman's rank correlation coefficient between the mean of the outputs and the target
    # This is done by ranking the data and then calculating Pearson's correlation
    rank_outputs = torch.argsort(torch.argsort(mean_outputs))
    rank_target = torch.argsort(torch.argsort(target.squeeze()))
    rank_ic = torch.corrcoef(torch.stack((rank_outputs.float(), rank_target.float())))[0, 1]

    return ic.item(), rank_ic.item()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    setup_seed(1129)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 配置日志输出到文件
    # localtime = time.localtime(time.time())
    # time1 = time.strftime("%m%d_%H%M",time.localtime(time.time())) 
    file_handler = logging.FileHandler('/home/hbliu/TCN_quant/log/train_{}.log'.format(time1), mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 获取根日志器，并添加上面定义的两个处理器
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)


    logging.info("reading data ... ...")

    df_data = pd.read_csv("/home/hbliu/tcn_data.csv")
    df_data = df_data.fillna(0.)
    
    time_points = df_data["datetime"].unique()
    fields = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'total_turnover']
    dict_data = {f:df_data.pivot_table(index='datetime', columns='instrument', values=f) for f in fields} # 每一天 每一支股票的属性

    input_channel = 7
    output_channel = 64
    num_channels = [10, 10, 20, 20, 10]
    model = TCN(input_channel, output_channel, num_channels)
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    lambda1 = 0.5
    lambda2 = 1

    iteration = 0
    year = 2008
    best_ic = 0
    while year < 2018:
        logging.info("loading train/valid/test set ... ... ")
        train_data = TCN_data(dict_data, time_points, year, "train")
        valid_data = TCN_data(dict_data, time_points, year, "valid")
        test_data = TCN_data(dict_data, time_points, year, "test")
        
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=16)
        valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)  
        
        reserve = False
        if reserve:
            logging.info("Loading checkpoint ... ...")
            saved_dict = torch.load("/home/hbliu/TCN_quant/ckpt/epoch12.bin")
            num_epoch = saved_dict['epoch']
            state_dict = saved_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            num_epoch = 0
            
        for epoch in range(num_epoch, num_epochs):  # 假设训练100个epoch
            train_loss = train(model, train_loader, epoch, optimizer, lambda1, lambda2)
            writer.add_scalar("Train loss", train_loss, iteration*num_epochs+epoch)
            logging.info(f"Training loss in iteration {iteration} epoch {epoch} is {train_loss}")
            # 可以添加验证步骤和保存模型的代码
            mean_ic, mean_rank_ic, eval_loss = evaluate_on_validation_dataset(model=model, validation_dataloader=valid_loader)
            logging.info(f"Iteration {iteration} Epoch {epoch} Mean IC: {mean_ic}, Mean Rank IC: {mean_rank_ic}, Eval loss {eval_loss}")
            writer.add_scalar("valid IC", mean_ic, iteration*num_epochs+epoch)
            writer.add_scalar("valid RankIC", mean_rank_ic, iteration*num_epochs+epoch)
            writer.add_scalar("valid loss", eval_loss, iteration*num_epochs+epoch)
            if epoch % 5 == 0:
                torch.save({"state_dict":model.state_dict(), "epoch":epoch, "iteration":iteration}, f'/home/hbliu/TCN_quant/ckpt/iteration{iteration}epoch{epoch}.bin')
            if mean_ic > best_ic:
                best_ic = mean_ic
                mean_ic_test, mean_rank_ic_test, eval_loss_test = evaluate_on_validation_dataset(model=model, validation_dataloader=test_loader)
                writer.add_scalar("Test IC", mean_ic_test, iteration*num_epochs+epoch)
                writer.add_scalar("Test RankIC", mean_rank_ic_test, iteration*num_epochs+epoch)
                writer.add_scalar("Test loss", eval_loss_test, iteration*num_epochs+epoch)
                logging.info("saving best validation model ... ...")
                torch.save({"state_dict":model.state_dict(), "epoch":epoch, "iteration":iteration}, '/home/hbliu/TCN_quant/ckpt/best_model.bin')
                logging.info(f"Iteration {iteration} Epoch {epoch} Test Mean IC: {mean_ic_test}, Test Mean Rank IC: {mean_rank_ic_test}, Test Eval loss {eval_loss_test}")
        
        year += 1
        iteration += 1        