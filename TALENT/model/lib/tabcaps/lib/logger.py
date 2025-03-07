import os
import torch
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter

class Train_Log():
    def __init__(self, logname, resume_dir=None, cfg=None):
        print('===> Setting log ...')
        time_str = datetime.now().strftime("%m-%d_%H%M")
        if resume_dir:
            self.resume_dir = os.path.join('./logs', resume_dir)
            self.log_dir = self.resume_dir

        else:
            self.log_dir = os.path.join('./logs/',  logname + '_' + time_str)

        # self.writer = SummaryWriter(self.log_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # self.save_config(cfg)

    def load_checkpoint(self, optimizer):
        lastest_out_path = "{}/checkpoint.pth".format(self.resume_dir)
        ckpt = torch.load(lastest_out_path)
        model = ckpt['model']
        start_epoch = ckpt['epoch'] + 1
        # model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_value = ckpt['best_value']
        best_epoch = ckpt['best_epoch']

        print("=> loaded checkpoint '{}' (epoch {})".format(lastest_out_path, ckpt['epoch']))

        return start_epoch, model, optimizer, best_value, best_epoch

    def save_best_model(self, model):
        lastest_out_path = self.log_dir + '/' + 'best' + '.pth'
        torch.save(model, lastest_out_path)
        print('====>Save Best model!!<=====')

    def save_log(self, log):
        mode = 'a' if os.path.exists(self.log_dir + '/log.txt') else 'w'
        logFile = open(self.log_dir + '/log.txt', mode)
        logFile.write(log + '\n')
        logFile.close()

    def save_config(self, cfg):
        with open(self.log_dir + '/config.yml', 'w') as f:
            f.write(cfg.dump())
        f.close()

    # def save_tensorboard(self, info, epoch):
    #     for tag, value in info.items():
    #         self.writer.add_scalar(tag, value, global_step=epoch)
