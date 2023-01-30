from absl import app, flags

from learner import Learner

flags.DEFINE_bool('proj_pre', False, 'proj_pre')
flags.DEFINE_bool('gsc', False, 'GSC')
flags.DEFINE_float('gsc_q', 0.7, 'GSC parameter q')
flags.DEFINE_string('proj_opt', 'sgd', 'proj_opt')
flags.DEFINE_float('proj_lr', 1e-3, 'learning rate')
flags.DEFINE_float('tau', 0.2, 'tau')
flags.DEFINE_float('a', 1.0, 'a')
flags.DEFINE_float('b', 1.0, 'b')
flags.DEFINE_integer('mini_batch_size', 256, 'mini_batch_size')
flags.DEFINE_integer('gpu', 0, 'gpu')
flags.DEFINE_bool('mix', False, 'Use mixup')
flags.DEFINE_bool('naive', False, 'naive mixup')
flags.DEFINE_bool('lisa', False, 'lisa')
flags.DEFINE_bool('ours', False, 'ours')
flags.DEFINE_bool('rand', False, 'rand')
flags.DEFINE_bool('gt', False, 'Use ground-truth label')

flags.DEFINE_bool('intra', False, 'contradicting positives')
flags.DEFINE_bool('inter', False, 'contradicting negatives')

flags.DEFINE_integer('batch_size', 256, 'batch_size')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('weight_decay', 0.0, 'weight_decay')
flags.DEFINE_float('momentum', 0.9, 'momentum')
flags.DEFINE_integer('num_workers', 4, 'workers number')
flags.DEFINE_string('exp', 'debugging', 'experiment name')
flags.DEFINE_string('device', 'cuda', 'cuda or cpu')
flags.DEFINE_integer('num_steps', 500 * 100, '# of iterations')
flags.DEFINE_integer('target_attr_idx', 0, 'target_attr_idx')
flags.DEFINE_integer('bias_attr_idx', 1, 'bias_attr_idx')
flags.DEFINE_string('dataset', 'cmnist', 'data to train, [cmnist, cifar10, bffhq]')
flags.DEFINE_string('percent', '1pct', 'percentage of conflict')
flags.DEFINE_float('q', 0.7, 'GCE parameter q')
flags.DEFINE_float('ema_alpha', 0.7, 'use weight mul')
flags.DEFINE_bool('use_type0', False, 'whether to use type 0 CIFAR10C')
flags.DEFINE_bool('use_type1', False, 'whether to use type 1 CIFAR10C')
flags.DEFINE_string('model', 'MLP', 'which network, [MLP, ResNet18, ResNet20, ResNet50]')

# logging
flags.DEFINE_string('log_path', '/data/bias/log/dfa', 'path for saving model')
flags.DEFINE_string('data_dir', '/data/bias/data', 'path for loading data')
flags.DEFINE_integer('valid_freq', 500, 'frequency to evaluate on valid/test set')
flags.DEFINE_integer('log_freq', 500, 'frequency to log on tensorboard')
flags.DEFINE_integer('save_freq', 1000, 'frequency to save model checkpoint')
flags.DEFINE_bool('wandb', True, 'whether to use wandb')
flags.DEFINE_bool('tensorboard', True, 'whether to use tensorboard')

# experiment
flags.DEFINE_bool('train_vanilla', False, 'Vanilla + Ours')
flags.DEFINE_bool('train_lff', False, 'LfF + Ours')


args = flags.FLAGS

# actual training
print('Official Pytorch Code of "SelecMix: Debiased Learning by Contradicting-pair Sampling (NeurIPS 2022)"')
print('Training starts ...')
def main(_):
    learner = Learner(args)
    if args.train_vanilla:
        learner.train_vanilla(args)
    elif args.train_lff:
        learner.train_lff(args)
    else:
        print('choose one of the two options ...')
        import sys
        sys.exit(0)

if __name__ == '__main__':
    app.run(main)
