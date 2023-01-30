'''Modified from https://github.com/alinlab/LfF and https://github.com/kakaoenterprise/Learning-Debiased-Disentangled'''

from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import \
    BatchSampler, SubsetRandomSampler

import os

from data.util import get_dataset, IdxDataset
from module.loss import GeneralizedCELoss, SupConLoss
from module.util import get_model
from module.resnet_big import SupConResNet
from util import EMA, setup_step_num, mixup_naive, mixup_with_gt_bias_label, mixup_with_similarity_matrix
import time


class Learner(object):
    def __init__(self, args):
        data2model = {'cmnist': "MLP",
                       'cifar10c': "ResNet18",
                       'bffhq': "ResNet18"}

        data2mini_batch_size = {'cmnist': 256,
                                'cifar10c': 256,
                                'bffhq': 64}

        data2batch_size = {'cmnist': 1024,
                           'cifar10c': 1024,
                           'bffhq': 256}
        
        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True}

        self.model = data2model[args.dataset]
        args.batch_size = data2batch_size[args.dataset]
        self.batch_size = args.batch_size
        args.mini_batch_size = data2mini_batch_size[args.dataset]
        self.mini_batch_size = args.mini_batch_size
        print ("batch_size: {}, {}".format(self.batch_size, args.batch_size))
        print ("mini_batch_size: {}, {}".format(self.mini_batch_size, args.mini_batch_size))

        args.num_steps = 15000
        args.valid_freq = 125
        args.log_freq = 125

        if args.wandb:
            import wandb
            wandb.init(project='SelecMix_{}_{}'.format(args.dataset, args.percent))
            wandb.run.name = args.exp

        run_name = args.exp + time.strftime("_%m%d_%H-%M-%S")
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_path, args.dataset, args.exp), exist_ok=True)
        self.device = f'cuda:{args.gpu}'
        self.args = args

        # logging directories
        self.log_dir = os.path.join(args.log_path, args.dataset, args.exp)
        self.summary_dir =  os.path.join(args.log_path, args.dataset, "summary", args.exp)
        self.summary_gradient_dir = os.path.join(self.log_dir, "gradient")
        self.result_dir = os.path.join(self.log_dir, "result")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_dataset = get_dataset(
            args,
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="train",
            transform_split="train",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )
        self.valid_dataset = get_dataset(
            args,
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="valid",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        self.test_dataset = get_dataset(
            args,
            args.dataset,
            data_dir=args.data_dir,
            dataset_split="test",
            transform_split="valid",
            percent=args.percent,
            use_preprocess=data2preprocess[args.dataset],
            use_type0=args.use_type0,
            use_type1=args.use_type1
        )

        train_target_attr = []
        for data in self.train_dataset.data:
            train_target_attr.append(int(data.split('_')[-2]))
        train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        self.train_dataset = IdxDataset(self.train_dataset)

        # make loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.mini_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.mini_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # define model and optimizer
        self.model_b = get_model(self.model, self.num_classes)
        self.model_d = get_model(self.model, self.num_classes)

        self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        self.optimizer_d = torch.optim.Adam(
                self.model_d.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        # define loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.bias_criterion = nn.CrossEntropyLoss(reduction='none')

        print(f'self.criterion: {self.criterion}')
        print(f'self.bias_criterion: {self.bias_criterion}')

        self.sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr).to(self.device), num_classes=self.num_classes, alpha=args.ema_alpha)
        self.sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr).to(self.device), num_classes=self.num_classes, alpha=args.ema_alpha)

        print(f'alpha : {self.sample_loss_ema_d.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.
        print('finished model initialization....')

    # evaluation code for ours
    def evaluate_ours(self, args, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0

        for data, attr, index in tqdm(data_loader, leave=False):
            label = attr[:, args.target_attr_idx]
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()
        return accs
    
    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_d.th")
        else:
            model_path = os.path.join(self.result_dir, "model_d_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def board_ours_loss(self, step, loss_d, loss_b, loss_d_mix):

        if self.args.wandb:
            wandb.log({
                "loss_d_mix":    loss_d_mix,
                "loss_d":    loss_d,
                "loss_b":       loss_b,
                "loss":                 (loss_d + loss_b)
            }, step=step,)

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_d",  loss_d, step)
            self.writer.add_scalar(f"loss/loss_b",     loss_b, step)
            self.writer.add_scalar(f"loss/loss",               (loss_d + loss_b), step)

    def board_ours_acc(self, args, step, epoch, inference=None):
        # check label network
        valid_accs_b = self.evaluate_ours(args, self.model_b, self.valid_loader)
        test_accs_b = self.evaluate_ours(args, self.model_b, self.test_loader)
        valid_accs_d = self.evaluate_ours(args, self.model_d, self.valid_loader)
        test_accs_d = self.evaluate_ours(args, self.model_d, self.test_loader)
        if inference:
            print(f'test acc: {test_accs_d.item()}')
            import sys
            sys.exit(0)

        print(f'epoch: {epoch}')
        wandb.log({'epoch': epoch}, step=step, )

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b
        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b

        # if valid_accs_d >= self.best_valid_acc_d:
        #     self.best_valid_acc_d = valid_accs_d

        if valid_accs_d > self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d
        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            # self.save_ours(step, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_b_valid": valid_accs_b,
                "acc_b_test": test_accs_b,
                "acc_d_valid": valid_accs_d,
                "acc_d_test": test_accs_d,
            },
                step=step, )
            wandb.log({
                "best_acc_b_valid": self.best_valid_acc_b,
                "best_acc_b_test": self.best_test_acc_b,
                "best_acc_d_valid": self.best_valid_acc_d,
                "best_acc_d_test": self.best_test_acc_d,
            },
                step=step, )

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, step)
            self.writer.add_scalar(f"acc/acc_b_test", test_accs_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)

        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b} ')
        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} ')
    
    def setup_auxiliary_contrastive_model(self, args):
        self.model_c = SupConResNet(name='resnet18').to(self.device)
        if args.proj_opt == "sgd":
            self.optimizer_proj = torch.optim.SGD(self.model_c.parameters(), lr=args.proj_lr, momentum=0.9, weight_decay=1e-4)
        elif args.proj_opt == "adam":
            self.optimizer_proj = torch.optim.Adam(self.model_c.parameters(), lr=args.proj_lr, weight_decay=1e-4)
        print ("args.gsc: {}".format(args.gsc))
        self.proj_criterion = SupConLoss(temp=args.tau, q=args.gsc_q, GSC=args.gsc)

    def train_lff(self, args):
        train_num = len(self.train_dataset)
        epch = int(train_num/args.batch_size)
        setup_step_num(args, epch)
        epoch, cnt = 0, 0

        if args.dataset == 'cmnist':
            self.model_d = get_model('MLP', self.num_classes).to(self.device)
            self.model_b = get_model('MLP', self.num_classes).to(self.device)
        else:
            self.model_d = get_model('ResNet18', self.num_classes).to(self.device)
            self.model_b = get_model('ResNet18', self.num_classes).to(self.device)

        self.optimizer_d = torch.optim.Adam(
            self.model_d.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.optimizer_b = torch.optim.Adam(
            self.model_b.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        if args.ours: self.setup_auxiliary_contrastive_model(args)        

        self.bias_criterion = GeneralizedCELoss(q=0.7)

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        train_iter = iter(self.train_loader)
        
        # custom
        mix_loss_d = 0

        for step in tqdm(range(args.num_steps)):
            if step == 0: print (">>> Step = 0 <<<")

            try:
                batch_index, batch_data, batch_attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                batch_index, batch_data, batch_attr, _ = next(train_iter)

            batch_data = batch_data.to(self.device)
            batch_attr = batch_attr.to(self.device)
            batch_label = batch_attr[:, args.target_attr_idx].to(self.device)

            with torch.no_grad():
                if args.mix :
                    if args.naive:
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_naive(batch_data, batch_label)
                    elif args.gt:
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_with_gt_bias_label(args, batch_data, batch_attr, batch_label)
                    elif args.ours and not args.gt:
                        sim = self.model_c.sim_matrix(batch_data)
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_with_similarity_matrix(args, batch_data, batch_label, sim)
                    f = batch_lam_a + batch_lam_b
                    batch_is_mix = (f != 0).long()

            sampler = BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                drop_last=True)

            for indices in sampler:

                data = batch_data[indices]
                label = batch_label[indices]
                index = batch_index[indices]

                logit_d = self.model_d(data)
                logit_b = self.model_b(data)

                loss_d = self.criterion(logit_d, label).detach()
                loss_b = self.criterion(logit_b, label).detach()

                # EMA sample loss
                self.sample_loss_ema_d.update(loss_d, index)
                self.sample_loss_ema_b.update(loss_b, index)

                # class-wise normalize
                loss_d = self.sample_loss_ema_d.parameter[index].clone().detach()
                loss_b = self.sample_loss_ema_b.parameter[index].clone().detach()

                loss_d = loss_d.to(self.device)
                loss_b = loss_b.to(self.device)

                for c in range(self.num_classes):
                    class_index = torch.where(label == c)[0].to(self.device)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    loss_d[class_index] /= max_loss_d
                    loss_b[class_index] /= max_loss_b

                loss_weight = loss_b / (loss_b + loss_d + 1e-8)                          # Eq.1 (reweighting module) in LfF paper
                loss_d_update = self.criterion(logit_d, label) * loss_weight.to(self.device)              # Eq.2 W(z)CE(C_i(z),y) in LfF paper
                loss_b_update = self.bias_criterion(logit_b, label)                                             # Eq.2 GCE(C_b(z),y) in LfF paper

                if args.mix:
                    mixed_data = mixed_batch_data[indices]
                    mix_label = mix_batch_label[indices]
                    lam_a = batch_lam_a[indices]
                    lam_b = batch_lam_b[indices]
                    is_mix = batch_is_mix[indices]

                    mix_logit_d = self.model_d(mixed_data)
                    mix_loss_d_update = lam_a * self.criterion(mix_logit_d, label) + lam_b * self.criterion(mix_logit_d, mix_label)
                    mix_loss_d = (mix_loss_d_update).sum() / is_mix.sum()
                    if args.ours: assert (lam_a.sum() == 0)
                
                loss  = args.b * loss_d_update.mean() + loss_b_update.mean()
                loss = loss + args.a * mix_loss_d

                self.optimizer_d.zero_grad()
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_d.step()
                self.optimizer_b.step()

                if args.ours and not args.gt:
                    z = self.model_c(data)
                    proj_loss = self.proj_criterion(z, label)
                    self.optimizer_proj.zero_grad()
                    proj_loss.backward()
                    self.optimizer_proj.step()

            # if (step + 1) % args.save_freq == 0:
            #     self.save_ours(step)

            if (step + 1) % args.log_freq == 0:
                self.board_ours_loss(
                    step=step,
                    loss_d=loss_d.mean(),
                    loss_b=loss_b.mean(),
                    loss_d_mix = mix_loss_d,
                )

            cnt += len(batch_index)
            if cnt + len(batch_index) > train_num:
                epoch += 1
                print(f'finished epoch: {epoch}, step: {step}, cnt: {cnt}')
                cnt = 0

            if (step + 1) % args.valid_freq == 0:
                with torch.no_grad(): self.board_ours_acc(args, step, epoch)

    def train_vanilla(self, args):
        train_num = len(self.train_dataset)
        epch = int(train_num/args.batch_size)
        setup_step_num(args, epch)
        epoch, cnt = 0, 0

        if args.dataset == 'cmnist':
            self.model_d = get_model('MLP', self.num_classes).to(self.device)
        else:
            self.model_d = get_model('ResNet18', self.num_classes).to(self.device)

        self.optimizer_d = torch.optim.Adam(
            self.model_d.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.ours: self.setup_auxiliary_contrastive_model(args)

        print(f'criterion: {self.criterion}')
        train_iter = iter(self.train_loader)
        loss_d = 0
        mix_loss_d =0

        for step in tqdm(range(args.num_steps)):
            if step == 0: print (">>> Step = 0 <<<")

            try:
                batch_index, batch_data, batch_attr, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                batch_index, batch_data, batch_attr, _ = next(train_iter)

            batch_data = batch_data.to(self.device)
            batch_attr = batch_attr.to(self.device)
            batch_label = batch_attr[:, args.target_attr_idx].to(self.device)

            with torch.no_grad():
                if args.mix:
                    if args.naive:
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_naive(batch_data, batch_label)
                    elif args.gt:
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_with_gt_bias_label(args, batch_data, batch_attr, batch_label)
                    elif args.ours and not args.gt:
                        sim = self.model_c.sim_matrix(batch_data)
                        mixed_batch_data, mix_batch_label, batch_lam_a, batch_lam_b = mixup_with_similarity_matrix(args, batch_data, batch_label, sim)

            sampler = BatchSampler(
                SubsetRandomSampler(range(self.batch_size)),
                self.mini_batch_size,
                drop_last=True)

            for indices in sampler:
                data = batch_data[indices]
                label = batch_label[indices]

                logit_d = self.model_d(data)
                loss_d = 0
                loss_d_update = self.criterion(logit_d, label)
                loss_d = loss_d_update.mean() 
                if args.mix:
                    mixed_data = mixed_batch_data[indices]
                    mix_label = mix_batch_label[indices]
                    lam_a = batch_lam_a[indices]
                    lam_b = batch_lam_b[indices]
                    mix_logit_d = self.model_d(mixed_data)
                    mix_loss_d_update = lam_a * self.criterion(mix_logit_d, label) + lam_b * self.criterion(mix_logit_d, mix_label)
                    mix_loss_d = mix_loss_d_update.mean()
                    if args.ours: assert (lam_a.sum() == 0)
                loss  = args.b * loss_d + args.a * mix_loss_d
                
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_d.step()
                if args.ours:
                    z = self.model_c(data)
                    proj_loss = self.proj_criterion(z, label)
                    self.optimizer_proj.zero_grad()
                    proj_loss.backward()
                    self.optimizer_proj.step()

            # if (step + 1) % args.save_freq == 0:
            #     self.save_vanilla(step, epoch)

            if (step + 1) % args.log_freq == 0:
                self.board_vanilla_loss(
                    step=step,
                    loss_d=loss_d,
                    loss_d_mix = mix_loss_d
                )

            cnt += len(batch_index)
            if cnt + len(batch_index) > train_num:
                epoch += 1
                print(f'finished epoch: {epoch}, step: {step}, cnt: {cnt}')
                cnt = 0

            if (step + 1) % args.valid_freq == 0:
                with torch.no_grad(): self.board_vanilla_acc(args, step, epoch)

    def board_vanilla_loss(self, step, loss_d, loss_d_mix):
        if self.args.wandb:
            wandb.log({
                "loss_d": loss_d,
                "loss_d_mix":    loss_d_mix,
            }, step=step,)

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_d", loss_d, step)

    def board_vanilla_acc(self, args, step, epoch, inference=None):
        # check label network
        valid_accs_d = self.evaluate_ours(args, self.model_d, self.valid_loader)
        test_accs_d = self.evaluate_ours(args, self.model_d, self.test_loader)

        print(f'epoch: {epoch}')
        wandb.log({'epoch': epoch}, step=step, )

        if valid_accs_d > self.best_valid_acc_d:
            self.best_valid_acc_d = valid_accs_d
        if test_accs_d >= self.best_test_acc_d:
            self.best_test_acc_d = test_accs_d
            # self.save_vanilla(step, epoch, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_d_valid": valid_accs_d,
                "acc_d_test": test_accs_d,
            },
                step=step, )
            wandb.log({
                "best_acc_d_valid": self.best_valid_acc_d,
                "best_acc_d_test": self.best_test_acc_d,
            },
                step=step, )

        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d}')

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
            self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)

            self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
            self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)
    
    def save_vanilla(self, step, epoch, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(step))
        state_dict = {
            'steps': step,
            'epoch': epoch,
            'state_dict': self.model_d.state_dict(),
            'optimizer': self.optimizer_d.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{step} model saved ...')
