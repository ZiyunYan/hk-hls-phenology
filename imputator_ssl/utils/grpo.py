import torch
import torch.nn as nn
import torch.nn.functional as F
from jinja2.utils import missing


def compute_rewards(reconstructed_ori, reconstructed_new, batch, missing_mask, anomalies_mask):
    # reconstructed_ori, batch: (batchsize, steps, bands)
    # missing_mask: (batchsize, steps, bands)
    # anomalies_mask: (n, batchsize, steps)
    # reconstructed_new: (n, batchsize, steps, bands)

    batch_size, steps, bands = batch.shape
    n = anomalies_mask.shape[0]

    # 计算重建误差
    ori_errors = torch.abs(reconstructed_ori - batch)  # (batchsize, steps, bands)
    new_errors = torch.abs(reconstructed_new - batch.unsqueeze(0))  # (n, batchsize, steps, bands)

    # 计算平滑度（使用一阶差分）
    ori_smoothness = torch.abs(reconstructed_ori[:, 1:] - reconstructed_ori[:, :-1]).mean(dim=(1, 2))  # (batchsize,)
    new_smoothness = torch.abs(reconstructed_new[:, :, 1:] - reconstructed_new[:, :, :-1]).mean(
        dim=(2, 3))  # (n, batchsize)

    rewards = []
    for i in range(n):
        # 计算异常标注比例
        if torch.sum(missing_mask[:,:,0].unsqueeze(-1) * (1 - anomalies_mask[i].unsqueeze(-1)))==0:
            reconstruction_reward = 0
            smoothness_reward = 0
        else:
            anomaly_ratio = anomalies_mask[i].float().mean()
            # 计算重建误差reward
            new_errors_mask = missing_mask[:,:,0].unsqueeze(-1) * (1 - anomalies_mask[i].unsqueeze(-1))  # (batchsize, steps, bands)
            ori_valid_errors = torch.sum(ori_errors * new_errors_mask)
            new_valid_error = torch.sum(new_errors[i] * new_errors_mask)
            reconstruction_reward = ori_valid_errors - new_valid_error

            # 计算平滑度reward
            # smoothness_reward = (ori_smoothness - new_smoothness[i]).mean() * 100  # 调整权重

            # # 计算异常标注惩罚
            # anomaly_penalty = torch.where(anomaly_ratio > 0.1,
            #                               (anomaly_ratio - 0.1) * 10,  # 大幅惩罚
            #                               anomaly_ratio)  # 小幅惩罚

            # 合并rewards
        total_reward = reconstruction_reward
        rewards.append(total_reward)

    rewards = torch.stack(rewards, dim=0)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    print("anomalies:", torch.sum(missing_mask[:,:,0].unsqueeze(-1) * (1 - anomalies_mask[0].unsqueeze(-1))))
    print("rewards:", rewards)
    return rewards, mean_reward, std_reward


class Anomalies_Loss(nn.Module):
    def __init__(self, coef=0.1, epsilon=0.2):
        super(Anomalies_Loss, self).__init__()
        self.epsilon = epsilon  # Clipping parameter
        self.coef = coef

    def forward(self, reconstructed_ori, reconstructed_new, active_probs, probs, rewards, mean, std):
        # active_probs: (batchsize, steps)
        # probs: (batchsize, steps)
        # advantages: (n, batchsize, steps)
        n, batch, steps, bands = reconstructed_new.shape
        cosine_similarities = []
        probs = probs + 1e-8
        std = std + 1e-8
        clipped_advantages = []
        for i in range(n):
            current_sequence = reconstructed_new[i]  # 维度: (batchsize, steps, bands)
            ratio = active_probs/probs
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            clipped_advantage = clipped_ratio*((rewards[i]-mean)/std)
            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(reconstructed_ori, current_sequence, dim=-1)
            cosine_similarities.append(cosine_sim)
            clipped_advantages.append(clipped_advantage)

        normalized_similarity = (torch.mean(torch.stack(cosine_similarities))+1)/2
        normalized_advantages = torch.mean(torch.stack(clipped_advantages))

        return -normalized_advantages - self.coef*normalized_similarity

    # def reinforce_train(self, setting):
    #     train_data, train_loader = self._get_data(flag='train')
    #     vali_data, vali_loader = self._get_data(flag='val')
    #     test_data, test_loader = self._get_data(flag='test')
    #
    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #
    #     time_now = time.time()
    #
    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=500, verbose=True)
    #
    #
    #     if self.args.pretrain_model is True:
    #         print(f"加载预训练模型")
    #         # 加载预训练模型参数
    #         checkpoint = torch.load(os.path.join('./checkpoints/' + setting + '-pretrain1', 'checkpoint.pth'))
    #         # 获取当前模型的 state_dict
    #         model_dict = self.model.state_dict()
    #         # 过滤掉形状不匹配的参数
    #         pretrained_dict = {k: v for k, v in checkpoint.items() if
    #                            k in model_dict and v.shape == model_dict[k].shape}
    #
    #         # 更新模型的参数
    #         model_dict.update(pretrained_dict)
    #         # 加载更新后的 state_dict
    #         self.model.load_state_dict(model_dict)
    #         for param in self.model.imputation_model.parameters():
    #             param.requires_grad = False
    #
    #
    #     model_optim = self._select_optimizer()
    #     # criterion = self._select_criterion()
    #     scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
    #                                         steps_per_epoch=train_steps,
    #                                         pct_start=self.args.pct_start,
    #                                         epochs=self.args.train_epochs,
    #                                         max_lr=self.args.learning_rate)
    #     anomaly_loss = Anomalies_Loss()
    #
    #     for epoch, (batch_x, start_mark, batch_x_mark, end_mark) in enumerate(train_loader):
    #         train_loss = []
    #         epoch_time = time.time()
    #         batch_x = batch_x.float().to(self.device)
    #         batch_x_mark = batch_x_mark.float().to(self.device)
    #         # 1. 创建有效数据掩码（非NaN的位置为1，NaN的位置为0）
    #         valid_mask = (1 - torch.isnan(batch_x).int()).to(self.device)
    #         batch_x = torch.nan_to_num(batch_x, nan=0.0)
    #         with torch.no_grad():
    #             outputs, atten, anomaly_prob, anomaly_mask = self.model(batch_x, batch_x_mark, valid_mask)
    #             new_outputs = []
    #             for i in range(len(anomaly_mask)):
    #                 new_batch_x = batch_x * (1 - anomaly_mask[i]).unsqueeze(-1)
    #                 new_output, _, _, _ = self.model(new_batch_x, batch_x_mark, valid_mask)
    #                 new_outputs.append(new_output)
    #             new_outputs = torch.stack(new_outputs, 0)
    #             rewards, re_mean, re_std = compute_rewards(outputs, new_outputs, batch_x, valid_mask, anomaly_mask)
    #
    #         self.model.train()
    #
    #         for i in range(self.args.train_epochs):
    #             model_optim.zero_grad()
    #             _, _, active_probs, _ = self.model(batch_x, batch_x_mark, valid_mask)
    #             loss = anomaly_loss(outputs, new_outputs, active_probs, anomaly_prob, rewards, re_mean, re_std)
    #             train_loss.append(loss.item())
    #             print("\tepoch: {0}, batch_iter: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #             loss.backward()
    #             model_optim.step()
    #             if self.args.lradj == 'TST':
    #                 adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
    #                 scheduler.step()
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #         early_stopping(train_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break
    #         if self.args.lradj != 'TST':
    #             adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
    #         else:
    #             print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    #
    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))
    #
    #     return self.model