import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kl_reward_window(original_seq, reconstructed_seq, window_size=15, epsilon=1e-8):
    """
    计算时序窗口的KL散度奖励
    输入形状：(batch_size, seq_len, num_bands)
    输出形状：(batch_size, seq_len)
    """
    batch_size, seq_len, num_bands = original_seq.shape
    padding = (window_size - 1) // 2  # 对称填充

    # 使用反射填充处理边界
    padded_original = F.pad(original_seq, (0, 0, padding, padding), mode='reflect')
    padded_reconstructed = F.pad(reconstructed_seq, (0, 0, padding, padding), mode='reflect')

    # 滑动窗口提取 (实现高效计算)
    windows_original = padded_original.unfold(1, window_size, 1)  # [B, L, C, W]
    windows_reconstructed = padded_reconstructed.unfold(1, window_size, 1)

    # 转换为概率分布（对每个窗口的每个波段）
    def to_distribution(windows):
        # 使用softmax归一化（可替换为其他归一化方法）
        probs = F.softmax(windows, dim=-1)  # [B, L, C, W]
        # 添加epsilon防止除零
        return probs + epsilon

    # 计算每个窗口的概率分布
    p = to_distribution(windows_original)
    q = to_distribution(windows_reconstructed)

    # 计算KL散度：sum(p * log(p/q))
    kl_div = p * (torch.log(p) - torch.log(q))  # [B, L, C, W]
    kl_div = kl_div.sum(dim=-1).sum(dim=-1)  # 对窗口和波段求和 [B, L]

    return kl_div

def compute_gae(gen_len, values, rewards, gamma=0.99, lam=0.9):
    """
    计算广义优势估计（GAE）

    参数:
    - gen_len: 时间步的数量（轨迹长度）
    - values: 每个时间步的状态值（张量形状为[batch_size, gen_len]）
    - rewards: 每个时间步的奖励（张量形状为[batch_size, gen_len]）
    - gamma: 折扣因子
    - lam: 优势估计中的lambda值

    返回:
    - advantages: 每个时间步的优势估计（张量形状为[batch_size, gen_len]）
    """
    device = values.device  # 获取values张量所在的设备

    advantages_reversed = []  # 用来存储反向计算的优势估计
    lastgaelam = torch.zeros(values.shape[0], device=device)  # 初始化最后的GAE并指定设备

    # 反向遍历时间步
    for t in reversed(range(gen_len)):
        # 获取下一个时间步的值，如果是最后一个时间步，则设置为0
        nextvalues = values[:, t + 1] if t < gen_len - 1 else torch.zeros_like(values[:, t], device=device)

        # 计算TD误差 delta
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]

        # 计算当前时间步的GAE（广义优势估计）
        lastgaelam = delta + gamma * lam * lastgaelam

        # 将计算的GAE保存到列表
        advantages_reversed.append(lastgaelam)

    # 将优势估计逆序堆叠，并转置得到正确的形状
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    return advantages


def compute_rewards(reconstructed_ori, reconstructed_new, batch, missing_mask, anomalies_mask):
    # all input shape (batchsize, steps, features)
    batch_size, steps, features = batch.shape
    missing_mask = missing_mask[:, :, 0]

    # 计算绝对误差
    ori_errors = torch.abs(reconstructed_ori - batch)
    new_errors = torch.abs(reconstructed_new - batch)

    # 应用掩码
    new_errors_mask = missing_mask.unsqueeze(-1) * (1 - anomalies_mask.unsqueeze(-1))

    ori_errors_masked = ori_errors * new_errors_mask
    new_errors_masked = new_errors * new_errors_mask

    local_rewards = compute_kl_reward_window(reconstructed_ori,
                                             reconstructed_new, window_size=15)

    # 将无效步的奖励设为0
    local_rewards = local_rewards * missing_mask

    # 计算总体奖励（与原来相同）
    ori_errors_sum = torch.sum(torch.sum(ori_errors_masked, dim=2), dim=1)
    new_errors_sum = torch.sum(torch.sum(new_errors_masked, dim=2), dim=1)


    total_rewards = ori_errors_sum - new_errors_sum

    # 将局部奖励乘以0.2，并在最后一步添加总体奖励
    scaled_local_rewards = -0.2 * local_rewards
    # print("scaled_local_rewards:", torch.sum(scaled_local_rewards))
    scaled_local_rewards[:, -1] += 10 * total_rewards
    print("total_rewards:", torch.sum(total_rewards))
    # print("ori_errors_sum:", ori_errors_sum)
    # print("new_errors_sum:", new_errors_sum)
    # print("ori_mean:", torch.mean(ori_mean))
    # print("new_mean:", torch.mean(new_mean))
    return scaled_local_rewards


class Anomalies_Loss(nn.Module):
    def __init__(self, value_loss_coef=0.5, epsilon=0.2):
        super(Anomalies_Loss, self).__init__()
        self.value_loss_coef = value_loss_coef
        self.epsilon = epsilon  # Clipping parameter

    def forward(self, active_probs, probs, advantages, returns, active_values):
        # 假设 active_probs 和 probs 的形状为 (batch_size, steps, 1)，表示动作 A 的概率
        # 对于动作 A，直接使用 active_probs 表示其概率
        probs = probs + 1e-8
        ratio = (active_probs / probs)
        ratio = ratio.squeeze(-1)

        # Clipping 操作
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        if torch.isnan(ratio).any() or torch.isinf(ratio).any():
            print("NaN or Inf in ratio")
        if torch.isnan(clipped_ratio).any() or torch.isinf(clipped_ratio).any():
            print("NaN or Inf in clipped_ratio")
        ppo_loss = torch.mean(-advantages * torch.min(ratio, clipped_ratio))
        # 计算 Value loss
        value_loss = torch.mean(0.5 * (returns - active_values) ** 2)
        # 总损失是两部分的加权和
        total_loss = ppo_loss + self.value_loss_coef * value_loss
        print("advantages:", advantages)
        print("ppo loss:", ppo_loss)
        print("value loss:", value_loss)
        return total_loss


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
    #         # 更新模型的参数
    #         model_dict.update(pretrained_dict)
    #         # 加载更新后的 state_dict
    #         self.model.load_state_dict(model_dict)
    #         for param in self.model.SaitsEmded1.parameters():
    #             param.requires_grad = False
    #         for param in self.model.encoder.parameters():
    #             param.requires_grad = False
    #         for param in self.model.SaitsEmded2.parameters():
    #             param.requires_grad = False
    #         for param in self.model.decoder.parameters():
    #             param.requires_grad = False
    #
    #
    #     model_optim = self._select_optimizer()
    #     # criterion = self._select_criterion()
    #
    #     scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
    #                                         steps_per_epoch=train_steps,
    #                                         pct_start=self.args.pct_start,
    #                                         epochs=self.args.train_epochs,
    #                                         max_lr=self.args.learning_rate)
    #
    #     anomaly_loss = Anomalies_Loss()
    #
    #
    #
    #     for epoch, (batch_x, start_mark, batch_x_mark, end_mark) in enumerate(train_loader):
    #         iter_count = 0
    #         train_loss = []
    #         self.model.eval()
    #         epoch_time = time.time()
    #         batch_x = batch_x.float().to(self.device)
    #         batch_x_mark = batch_x_mark.float().to(self.device)
    #         # 1. 创建有效数据掩码（非NaN的位置为1，NaN的位置为0）
    #         valid_mask = (1 - torch.isnan(batch_x).int()).to(self.device)
    #         batch_x = torch.nan_to_num(batch_x, nan=0.0)
    #         outputs, anomalies_prob, values, anomalies_mask = self.model(batch_x, batch_x_mark, None, None,
    #                                                                      valid_mask, -1)
    #         new_batch_x = batch_x * (1-anomalies_mask).unsqueeze(-1)
    #         if torch.equal(batch_x, new_batch_x):
    #             print("The inputs are exactly the same.")
    #         new_outputs, _, _, _ = self.model(new_batch_x, batch_x_mark, None, None,
    #                                                          valid_mask, -1)
    #         if torch.equal(outputs, new_outputs):
    #             print("The outputs are exactly the same.")
    #         rewards = compute_rewards(outputs, new_outputs, batch_x, valid_mask, anomalies_mask)
    #         advantages = compute_gae(366, values, rewards)
    #         returns = advantages + rewards
    #         # 分离张量以切断梯度
    #         anomalies_prob = anomalies_prob.detach()
    #         advantages = advantages.detach()
    #         returns = returns.detach()
    #         self.model.train()
    #         for i in range(self.args.train_epochs):
    #             model_optim.zero_grad()
    #             epoch_time = time.time()
    #             iter_count += 1
    #             _, active_anomalies_prob, active_values, new_anomalies_mask = self.model(batch_x, batch_x_mark, None, None,
    #                                                                          valid_mask, -1)
    #             # if torch.all(new_anomalies_mask == 0):
    #             #     print("anomalies_mask is all zeros")
    #             loss = anomaly_loss(active_anomalies_prob, anomalies_prob, advantages, returns, active_values)
    #
    #             train_loss.append(loss.item())
    #
    #
    #             print("\tepoch: {0}, batch_iter: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #
    #             loss.backward()
    #             model_optim.step()
    #
    #             if self.args.lradj == 'TST':
    #                 adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
    #                 scheduler.step()
    #
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         train_loss = np.average(train_loss)
    #
    #         early_stopping(train_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break
    #
    #         if self.args.lradj != 'TST':
    #             adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=True)
    #         else:
    #             print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
    #
    #
    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path))
    #
    #     return self.model