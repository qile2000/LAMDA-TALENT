from typing import Dict

from TALENT.model.lib.realmlp.training.scheduling import ConstantSchedule, get_schedule

# layers are created multiple times => either only register after stacking or allow to register multiple times


class HyperparamManager:
    class HyperGetter:
        def __init__(self, tc: 'HyperparamManager', hyper_name: str, base_value_pattern: str, sched_pattern: str):
            self.tc = tc
            self.hyper_name = hyper_name
            self.base_value_pattern = base_value_pattern
            self.sched_pattern = sched_pattern

        def __call__(self):
            return self.tc.hyper_base_values[self.hyper_name][self.base_value_pattern] * \
                   self.tc.get_hyper_sched_values()[self.hyper_name][self.sched_pattern]

    def __init__(self, **config):
        self.config = config
        self.hyper_base_values = {}
        self.hyper_scheds = {}
        self.hyper_sched_values = None
        # regularization terms
        self.reg_terms = []
        self.needs_update = True  # indicates whether self.hyper_sched_values needs to be updated
        self.more_info_dict = {}  # can be set from outside

    def get_more_info_dict(self) -> Dict:
        return self.more_info_dict

    def _find_pattern(self, d: dict, scope):
        pattern = None
        for key in d:
            if scope.matches(key):
                #print(d, scope, key)
                pattern = key
        if pattern is None:  # no pattern was found
            raise ValueError(f'No key in dict {d} matches scope {str(scope)}')
        return pattern

    def register_hyper(self, name: str, scope, default=None, default_sched=lambda: ConstantSchedule(1.0)):
        if name not in self.hyper_scheds:
            base_dict = self.config.get(name, default)
            if not isinstance(base_dict, dict):
                base_dict = {'': base_dict}
            sched_dict = self.config.get(name + '_sched', default_sched)
            if not isinstance(sched_dict, dict):
                sched_dict = {'': sched_dict}
            sched_dict = {key: get_schedule(sched) if isinstance(sched, str) else sched()
                            for key, sched in sched_dict.items()}
            self.hyper_scheds[name] = sched_dict
            self.hyper_base_values[name] = base_dict
            self.needs_update = True

        return HyperparamManager.HyperGetter(self, name,
                                            base_value_pattern=self._find_pattern(self.hyper_base_values[name], scope),
                                            sched_pattern=self._find_pattern(self.hyper_scheds[name], scope))

    # def _to_array(self, value, name: str, length: int) -> torch.Tensor:
    #     if hasattr(value, "__len__"):
    #         # result is already a list or a numpy array
    #         if len(value) != length:
    #             raise ValueError(f'Hyperparameter {name} has {len(value)} values but should have {length} values')
    #         return torch.as_tensor(value)
    #     else:
    #         return torch.as_tensor([value] * length)

    def get_hyper_sched_values(self):
        self.update_hyper_sched_values()
        return self.hyper_sched_values

    def update_hyper_sched_values(self):
        if self.needs_update:
            # print(f'update')
            self.hyper_sched_values = {name: {pattern: sched.get_value() for pattern, sched in sched_dict.items()}
                                       for name, sched_dict in self.hyper_scheds.items()}
            self.needs_update = False

    def add_reg_term(self, loss):
        self.reg_terms.append(loss)

    def update_hypers(self, learner):
        # reset regularization terms
        self.reg_terms = []

        self.needs_update = True

        for name, sched_dict in self.hyper_scheds.items():
            for pattern, sched in sched_dict.items():
                sched.update(learner)

        self.update_hyper_sched_values()




