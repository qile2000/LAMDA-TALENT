import torch
import torch.optim as optim

from TALENT.model.lib.realmlp.training.coord import HyperparamManager
from TALENT.model.lib.realmlp.optim.scheduling_adam import SchedulingAdam


class OptimizerBase(torch.optim.Optimizer):
    def __init__(self, opt, hyper_mappings, hp_manager: HyperparamManager):
        self.hp_manager = hp_manager
        self.hyper_getters = {}
        self.n_groups = len(opt.param_groups)
        for names, opt_name, defaults in hyper_mappings:
            if isinstance(names, str):
                names = (names,)
                defaults = (defaults,)
            for name, default in zip(names, defaults):
                self.hyper_getters[name] = [self.hp_manager.register_hyper(name, group['params'][0].context.scope,
                                                                           default=default) for group in opt.param_groups]
        super().__init__(opt.param_groups, defaults={})
        self.hyper_mappings = hyper_mappings
        self.opt = opt

    def get_hyper_values(self, name, i, use_hyper_factor=True):
        value = self.hyper_getters[name][i]()
        param = self.opt.param_groups[i]['params'][0]  # should only be one param
        if use_hyper_factor and name in param.hyper_factors:
            value *= param.hyper_factors[name]
        return value

    def step(self, closure=None):
        unhandled_mappings = []
        for names, opt_name, defaults in self.hyper_mappings:
            if opt_name is None:
                unhandled_mappings.append((names, opt_name, defaults))
                continue

            if isinstance(names, tuple):
                for i, group in enumerate(self.opt.param_groups):
                    group[opt_name] = tuple(self.get_hyper_values(name, i) for name in names)
            elif isinstance(names, str):
                for i, group in enumerate(self.opt.param_groups):
                    group[opt_name] = self.get_hyper_values(names, i)
            else:
                raise RuntimeError('Could not understand mapping key {}'.format(names))

        for names, opt_name, defaults in unhandled_mappings:
            if names == 'wd':
                with torch.no_grad():
                    for i, group in enumerate(self.opt.param_groups):
                        wd = self.get_hyper_values('wd', i)
                        lr = self.get_hyper_values('lr', i)
                        if wd != 0.0:
                            for p in group['params']:
                                p.mul_(1.0 - wd * lr * p.hyper_factors.get('wd', 1.0) * p.hyper_factors.get('lr', 1.0))

            else:
                raise RuntimeError('Could not understand mapping {}'.format((names, opt_name, defaults)))

        self.opt.step()


class AdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager):
        super().__init__(optim.Adam(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class SchedulingAdamOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager):
        super().__init__(SchedulingAdam(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class AMSGradOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager):
        super().__init__(optim.Adam(param_groups, amsgrad=True),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class AdamaxOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager):
        super().__init__(optim.Adamax(param_groups),
                         hyper_mappings=[('lr', 'lr', 1e-3), (('mom', 'sq_mom'), 'betas', (0.9, 0.999)),
                                         ('opt_eps', 'eps', 1e-8), ('wd', None, 0.0)],
                         hp_manager=hp_manager)


class SGDOptimizer(OptimizerBase):
    def __init__(self, param_groups, hp_manager):
        super().__init__(optim.SGD(param_groups), hyper_mappings=[('lr', 'lr', 1e-3), ('mom', 'momentum', 0.0),
                                                                  ('wd', None, 0.0)],
                         hp_manager=hp_manager)



def get_opt_class(opt_name):
    if opt_name == 'adam':
        return AdamOptimizer
    elif opt_name == 'adamax':
        return AdamaxOptimizer
    elif opt_name == 'sgd':
        return SGDOptimizer
    elif opt_name == 'amsgrad':
        return AMSGradOptimizer
    elif opt_name == 'sched_adam':
        return SchedulingAdamOptimizer
