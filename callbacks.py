import re
from utils import listify,partial,normalise,AvgStatsCap,AvgStats,cap_normalise,extract_mean_std
from torch import nn
import matplotlib.pyplot as plt
import torch
import wandb
from warnings import warn
import numpy as np
from pathlib import Path
import random
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
import time
from torch import tensor


imagenet_mean,imagenet_std = tensor([0.485, 0.4560, 0.4060]), tensor([0.2290, 0.2240, 0.2250]) 


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False
        self.run.in_train = False


class NewParamSchedulerCallback(Callback):
    def __init__(self,param,sched_funcs):
        self.param = param
        self.sf = listify(sched_funcs)

    def change_param(self):
        po = self.run.n_epochs/self.run.epochs
        fs = self.sf
        if len(fs) ==  1:
            fs = fs*len(self.opt.params)
        for f,i in zip(fs,self.opt.hypers):
            i[self.param] = f(po)

    def begin_batch(self):
        if self.in_train:
            self.change_param()



class NewRecorderCallback(Callback):
    def begin_fit(self):
        self.lrs, self.train_losses, self.valid_losses,self.train_acc,self.valid_acc = [],[],[],[],[]

    def after_loss(self):
        if self.in_train:
            self.lrs.append(self.opt.hypers[-1]['lr'])
            self.train_losses.append(self.loss.detach().cpu())
            self.train_acc.append(self.avg_stats.train_stats.avg_stats[-1])
        else:
            self.valid_losses.append(self.loss.detach().cpu())
            self.valid_acc.append(self.avg_stats.valid_stats.avg_stats[-1])

    def plot_lr(self):
        plt.plot(self.lrs)

    def plot_losses(self, loss_type = 'train_loss'):
        if loss_type == 'train_loss':
            plt.plot(self.train_losses)
            plt.ylabel('train_loss')
            plt.xlabel('Iterations')
        else:
            plt.plot(self.valid_losses)
            plt.ylabel('valid_loss')
            plt.xlabel('Iterations')

    def plot_acc(self,acc_type = 'train'):
        if acc_type == 'train':
            plt.plot(self.train_acc)
        else:
            plt.plot(self.valid_acc)


class NewLR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss


class CudaCapCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb, self.run.yb = (self.xb[0].cuda(), self.xb[1].cuda()),self.yb.cuda()

class AvgStatsCallback(Callback):
    def __init__(self, metrics,stats_collector = AvgStats):
        self.train_stats,self.valid_stats = stats_collector(metrics,True),stats_collector(metrics,False)
    
    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)
    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        stats = [str(self.epoch)] 
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats] 
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)

def normalise_callback(data,normalise_func = normalise, use_imagenet_stats = False):
    
    if use_imagenet_stats:
        m,s = imagenet_mean, imagenet_std
    else:
        x,_= next(iter(data.train_dl))    
        m,s = extract_mean_std(x[0]) if normalise_func == cap_normalise else extract_mean_std(x) 
    norm = partial(normalise_func, mean = m.cuda(), std = s.cuda())
    return partial(BatchTransformCallback, norm)

class BatchTransformCallback(Callback):
    _order = 2
    def __init__(self,tfm):self.f = tfm
    def begin_batch(self): self.run.xb = self.f(self.xb)


# class AvgStatsCaptionCallback(Callback):
#     def __init__(self, metrics):
#         self.train_stats,self.valid_stats = AvgStatsCap(metrics,True),AvgStatsCap(metrics,False)

#     def begin_epoch(self):
#         self.train_stats.reset()
#         self.valid_stats.reset()

#     def after_loss(self):
#         stats = self.train_stats if self.in_train else self.valid_stats
#         with torch.no_grad(): stats.accumulate(self.run)

#     def after_epoch(self):
#         #We use the logger function of the `Learner` here, it can be customized to write in a file or in a progress bar
#         self.logger(self.train_stats)
#         self.logger(self.valid_stats)

class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

class TrackerCallback(Callback):
    _order = 500
    "A `LearnerCallback` that keeps track of the best value in `monitor`."
    def __init__(self, monitor:str='valid_loss', mode:str='auto'):
        self.monitor,self.mode = monitor,mode
        if self.mode not in ['auto', 'min', 'max']:
            warn(f'{self.__class__} mode {self.mode} is invalid, falling back to "auto" mode.')
            self.mode = 'auto'
        mode_dict = {'min': np.less, 'max':np.greater}
        mode_dict['auto'] = np.less if 'loss' in self.monitor or 'error' in self.monitor else np.greater
        self.operator = mode_dict[self.mode]

    def begin_fit(self, **kwargs)->None:
        "Initializes the best value."
        self.best = float('inf') if self.operator == np.less else -float('inf')

    def get_monitor_value(self):
        "Pick the monitored value."
        if self.monitor=='train_loss' and len(self.new_recorder.train_losses) == 0: return None
        elif len(self.new_recorder.valid_losses) == 0: return None
        values = {'train_loss':self.new_recorder.train_losses[-1].cpu().numpy(),
                  'valid_loss':self.new_recorder.valid_losses[-1].cpu().numpy()}
        if values['valid_loss'] is None: return
        if self.avg_stats:
            for m, n in enumerate(self.avg_stats.train_stats.metrics):
                values['train_' + n.__name__] = self.avg_stats.train_stats.avg_stats[m+1]
            for m, n in enumerate(self.avg_stats.valid_stats.metrics):
                values['valid_' + n.__name__] = self.avg_stats.valid_stats.avg_stats[m+1]
        if values.get(self.monitor) is None:
            warn(f'{self.__class__} conditioned on metric `{self.monitor}` which is not available')
        return values.get(self.monitor)


#export
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model when monitored quantity is best."
    def __init__(self,monitor:str='valid_loss', mode:str='auto', every:str='improvement', name:str='bestmodel'):
        super().__init__(monitor=monitor, mode=mode)
        self.every,self.nam = every,name
        if self.every not in ['improvement', 'epoch']:
            warn(f'SaveModel every {self.every} is invalid, falling back to "improvement".')
            self.every = 'improvement'

    def jump_to_epoch(self, epoch:int)->None:
        try:
            self.learn.load(f'{self.nam}_{epoch-1}', purge=False)
            print(f"Loaded {self.nam}_{epoch-1}")
        except: print(f'Model {self.nam}_{epoch-1} not found.')

    def after_epoch(self):
        "Compare the value monitored to its best score and maybe save the model."
        if self.every=="epoch": self.run.save(f'{self.nam}_{self.epoch}')
        else: #every="improvement"
            current = self.get_monitor_value()
            if isinstance(current, torch.Tensor): current = current.cpu()
            if current is not None and self.operator(current, self.best):
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {current}.')
                self.best = current
                self.run.save(f'{self.nam}')



class WandbCallback(TrackerCallback):
    _order = 1000
    """
        Automatically saves model topology, losses & metrics.
        Optionally logs weights, gradients, sample predictions and best trained model.

        Args:
            learn (fastai.basic_train.Learner): the fast.ai learner to hook.
            log (str): "gradients", "parameters", "all", or None. Losses & metrics are always logged.
            save_model (bool): save model at the end of each epoch. It will also load best model at the end of training.
            monitor (str): metric to monitor for saving best model. None uses default TrackerCallback monitor value.
            mode (str): "auto", "min" or "max" to compare "monitor" values and define best model.
            input_type (str): "images" or None. Used to display sample predictions.
            validation_data (list): data used for sample predictions if input_type is set.
            predictions (int): number of predictions to make if input_type is set and validation_data is None.
            seed (int): initialize random generator for sample predictions if input_type is set and validation_data is None.
    """

    # Record if watch has been called previously (even in another instance)
    _watch_called = False

    def __init__(self,
                 log="gradients",
                 save_model=True,
                 monitor=None,
                 mode='auto',
                 input_type=None,
                 validation_data=None,
                 predictions=36,
                 seed=12345):

        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError(
                'You must call wandb.init() before WandbCallback()')

        # Adapted from fast.ai "SaveModelCallback"
        if monitor is None:
            # use default TrackerCallback monitor value
            super().__init__(mode=mode)
        else:
            super().__init__(monitor=monitor, mode=mode)
        self.save_model = save_model
        self.model_path = Path(wandb.run.dir) / 'bestmodel.pth'

        self.log = log
        self.input_type = input_type
#         self.best = None

        # Select items for sample predictions to see evolution along training
        self.validation_data = validation_data
        if input_type and not self.validation_data:
            wandbRandom = random.Random(seed)  # For repeatability
            predictions = min(predictions, len(self.data.valid_ds))
            indices = wandbRandom.sample(range(len(self.data.valid_ds)),
                                         predictions)
            self.validation_data = [self.data.valid_ds[i] for i in indices]

    def begin_fit(self):
        "Call watch method to log model topology, gradients & weights"

        # Set self.best, method inherited from "TrackerCallback" by "SaveModelCallback"
        super().begin_fit()
        
        key_dict = {k:self.new_recorder for k in list(self.new_recorder.__dict__.keys())[1:]}
#         key_dict2 = {k:self.avg_stats for k in list(self.avg_stats.__dict__.keys())[:-2]}
#         key_dict2.update(key_dict1)
        
        self.valid_dict = {k:v for k,v in key_dict.items() if 'valid' in k}
        self.train_dict = {k:v for k,v in key_dict.items() if k not in self.valid_dict}
        # Ensure we don't call "watch" multiple times
        if not WandbCallback._watch_called:
            WandbCallback._watch_called = True

            # Logs model topology and optionally gradients and weights
            wandb.watch(self.model, log=self.log)

    def after_epoch(self):
        "Logs training loss, validation loss and custom metrics & log prediction samples & save model"

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(
                    'Better model found at epoch {} with {} value: {}.'.format(
                        self.epoch, self.monitor, current))
                self.best = current

                # Save within wandb folder
                with self.model_path.open('wb') as model_file:
                    self.run.save(model_file)

        # Log sample predictions if learn.predict is available
        if self.validation_data:
            try:
                self._wandb_log_predictions()
            except FastaiError as e:
                wandb.termwarn(e.message)
                self.validation_data = None  # prevent from trying again on next loop
            except Exception as e:
                wandb.termwarn("Unable to log prediction samples.\n{}".format(e))
                self.validation_data=None  # prevent from trying again on next loop

        # Log losses & metrics
        # Adapted from fast.ai "CSVLogger"
  
    def after_loss(self): 
        if self.in_train:
            logs = {
                name: getattr(stat, name)[-1]
                for name, stat in self.train_dict.items()
            }
            wandb.log(logs)
            
        elif not self.in_train:
            logs = {
                name: getattr(stat, name)[-1]
                for name, stat in self.valid_dict.items()
            }
            wandb.log(logs)
            
    @staticmethod    
    def getatr(obj, values):
        ls = []
        for i in values:
            ls.append(getattr(obj, i)[-1])
        return tuple(ls)

    def after_fit(self, **kwargs):
        "Load the best model."

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            if self.model_path.is_file():
                with self.model_path.open('rb') as model_file:
                    self.run.load(model_file)
                    print('Loaded best saved model from {}'.format(
                        self.model_path))

    def _wandb_log_predictions(self):
        "Log prediction samples"

        pred_log = []

        for x, y in self.validation_data:
            try:
                pred=self.run.predict(x)
            except:
                raise FastaiError('Unable to run "predict" method from Learner to log prediction samples.')

            # scalar -> likely to be a category
            # tensor of dim 1 -> likely to be multicategory
            if not pred[1].shape or pred[1].dim() == 1:
                pred_log.append(
                    wandb.Image(
                        x.data,
                        caption='Ground Truth: {}\nPrediction: {}'.format(
                            y, pred[0])))

            # most vision datasets have a "show" function we can use
            elif hasattr(x, "show"):
                # log input data
                pred_log.append(
                    wandb.Image(x.data, caption='Input data', grouping=3))

                # log label and prediction
                for im, capt in ((pred[0], "Prediction"),
                                 (y, "Ground Truth")):
                    # Resize plot to image resolution
                    # from https://stackoverflow.com/a/13714915
                    my_dpi = 100
                    fig = plt.figure(frameon=False, dpi=my_dpi)
                    h, w = x.size
                    fig.set_size_inches(w / my_dpi, h / my_dpi)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    # Superpose label or prediction to input image
                    x.show(ax=ax, y=im)
                    pred_log.append(wandb.Image(fig, caption=capt))
                    plt.close(fig)

            # likely to be an image
            elif hasattr(y, "shape") and (
                (len(y.shape) == 2) or
                    (len(y.shape) == 3 and y.shape[0] in [1, 3, 4])):

                pred_log.extend([
                    wandb.Image(x.data, caption='Input data', grouping=3),
                    wandb.Image(pred[0].data, caption='Prediction'),
                    wandb.Image(y.data, caption='Ground Truth')
                ])

            # we just log input data
            else:
                pred_log.append(wandb.Image(x.data, caption='Input data'))

            wandb.log({"Prediction Samples": pred_log}, commit=False)


class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)