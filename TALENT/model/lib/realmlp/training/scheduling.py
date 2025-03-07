import numpy as np
import math


class LearnerProgress:
    def __init__(self):
        self.epoch = 0
        self.epoch_steps = 0
        self.total_steps = 0
        self.epoch_samples = 0
        self.total_samples = 0
        self.epoch_float = 0.0
        self.max_epochs = 0

    def get_fit_progress(self):
        return None if self.max_epochs is None else self.epoch_float / self.max_epochs


def sched_prod(first, second):
    if not isinstance(first, Schedule):
        first = ConstantSchedule(first)
    if not isinstance(second, Schedule):
        second = ConstantSchedule(second)
    if isinstance(first, TimeSchedule) and isinstance(second, TimeSchedule):
        return ProductTimeSchedule_(first, second)
    return ProductSchedule_(first, second)


def sched_sum(first, second):
    if not isinstance(first, Schedule):
        first = ConstantSchedule(first)
    if not isinstance(second, Schedule):
        second = ConstantSchedule(second)
    if isinstance(first, TimeSchedule) and isinstance(second, TimeSchedule):
        return SumTimeSchedule_(first, second)
    return SumSchedule_(first, second)


class Schedule:
    def get_value(self):
        raise NotImplementedError()

    def update(self, learner):
        raise NotImplementedError()

    def __mul__(self, other):
        return sched_prod(self, other)

    def __rmul__(self, other):
        return sched_prod(other, self)

    def __add__(self, other):
        return sched_sum(self, other)

    def __radd__(self, other):
        return sched_sum(other, self)

    def __neg__(self):
        return -1.0 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)


class TimeSchedule(Schedule):
    def __init__(self):
        self.t = 0.0

    def call_time_(self, t: float):
        raise NotImplementedError()

    def get_value(self):
        return self.call_time_(self.t)

    def update(self, learner):
        self.t = learner.progress.get_fit_progress()

    def scaled(self, ymin=0., ymax=1., tmin=0., tmax=1.):
        return ScaledSchedule(self, ymin, ymax, tmin, tmax)

    def reversed(self):
        return self.scaled(tmin=1., tmax=0.)


class ConstantSchedule(TimeSchedule):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def call_time_(self, t: float):
        return self.val


class FunctionSchedule(TimeSchedule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def call_time_(self, t: float):
        return self.f(t)


class ScaledSchedule(TimeSchedule):
    def __init__(self, base_schedule: TimeSchedule, ymin=0., ymax=1., tmin=0., tmax=1.):
        super().__init__()
        self.base_schedule = base_schedule
        self.ymin = ymin
        self.ymax = ymax
        self.tmin = tmin
        self.tmax = tmax

    def call_time_(self, t: float):
        return self.ymin + (self.ymax - self.ymin) * self.base_schedule.call_time_(
            self.tmin + (self.tmax - self.tmin) * t)


class ProductSchedule_(Schedule):
    def __init__(self, first: Schedule, second: Schedule):
        super().__init__()
        self.first = first
        self.second = second

    def get_value(self):
        return self.first.get_value() * self.second.get_value()

    def update(self, learner):
        self.first.update(learner)
        self.second.update(learner)


class ProductTimeSchedule_(TimeSchedule):
    def __init__(self, first: TimeSchedule, second: TimeSchedule):
        super().__init__()
        self.first = first
        self.second = second

    def call_time_(self, t: float):
        return self.first.call_time_(t) * self.second.call_time_(t)


class SumSchedule_(Schedule):
    def __init__(self, first: Schedule, second: Schedule):
        super().__init__()
        self.first = first
        self.second = second

    def get_value(self):
        return self.first.get_value() + self.second.get_value()

    def update(self, learner):
        self.first.update(learner)
        self.second.update(learner)


class SumTimeSchedule_(TimeSchedule):
    def __init__(self, first: TimeSchedule, second: TimeSchedule):
        super().__init__()
        self.first = first
        self.second = second

    def call_time_(self, t: float):
        return self.first.call_time_(t) + self.second.call_time_(t)


class ScheduleSequence(TimeSchedule):
    def __init__(self, lengths, schedules):
        super().__init__()
        self.lengths = np.array(lengths)
        self.event_times = np.hstack([[0.], np.cumsum(self.lengths)])
        self.schedules = schedules

    def call_time_(self, t: float):
        idx = np.max(np.argwhere(self.event_times <= t))
        idx = min(idx, len(self.schedules)-1)
        start = self.event_times[idx]
        end = self.event_times[idx+1]
        return self.schedules[idx].call_time_((t-start)/(end-start))


class ExponentialSchedule(TimeSchedule):
    def __init__(self, start, end):
        super().__init__()
        self.log_start = np.log(start)
        self.log_end = np.log(end)

    def call_time_(self, t: float):
        return np.exp(self.log_start + t * (self.log_end - self.log_start))


def cos_warm_func(x):
    if x < 2 ** (-10):
        return 1.0
    else:
        base_x = 2**(int(np.log2(x))-1)  # negative float values are rounded up
        return 0.5 + 0.5*np.cos(np.pi * (x/base_x - 1))


def combine_scheds(lengths, schedules):
    return ScheduleSequence(lengths, schedules)


def get_cos_sched() -> FunctionSchedule:
    return FunctionSchedule(lambda x: 0.5 * (1.0 - math.cos(math.pi * x)))


def get_id_sched() -> FunctionSchedule:
    return FunctionSchedule(lambda x: x)


def get_lin_sched() -> FunctionSchedule:
    return FunctionSchedule(lambda x: 1.-x)


def get_cos_warm_sched() -> FunctionSchedule:
    return FunctionSchedule(cos_warm_func)


def connect_cos_scheds(times, values):
    return combine_scheds([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])],
                          [get_cos_sched().scaled(v1, v2) for v1, v2 in zip(values[:-1], values[1:])])


def connect_lin_scheds(times, values):
    return combine_scheds([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])],
                          [get_cos_sched().scaled(v1, v2) for v1, v2 in zip(values[:-1], values[1:])])


class FirstToLastSchedule(TimeSchedule):
    def __init__(self, n_params):
        super().__init__()
        argmax_points = np.linspace(0.2, 0.6, n_params)
        self.scheds = [combine_scheds([t, 1.-t], [get_cos_sched().scaled(0.04, 1.), get_cos_sched().scaled(1., 1e-5)])
                  for t in argmax_points]

    def call_time_(self, t: float):
        return np.array([s.call_time_(t) for s in self.scheds])


class StepFunctionSchedule(Schedule):
    def __init__(self, f):
        self.step = 0
        self.f = f

    def update(self, learner):
        self.step = learner.progress.total_steps

    def get_value(self):
        return self.f(self.step)


class EpochLengthSqMomSchedule(Schedule):
    def __init__(self, min_value: float = 0.95, base_value: float = 0.5):
        self.value = min_value
        self.min_value = min_value
        self.base_value = base_value

    def update(self, learner):
        n_batches_per_epoch = len(learner.data_loader)
        self.value = max(self.min_value, self.base_value ** (1 / n_batches_per_epoch))

    def get_value(self):
        return self.value


class CoslogFunc:
    def __init__(self, n_cycles: int):
        self.n_cycles = n_cycles

    def __call__(self, t):
        return 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + (2 ** self.n_cycles - 1) * t)))


def cos_func(x):
    return 0.5 * (1.0 - math.cos(math.pi * x))


def identity_func(x):
    return x


def lin_func(x):
    return 1 - x


def get_schedule(sched_name: str):
    sched_type = sched_name
    base_sched = None

    cos_sched = FunctionSchedule(cos_func)
    # id_sched = FunctionSchedule(identity_func)
    lin_sched = FunctionSchedule(lin_func)
    cos_warm_sched = FunctionSchedule(cos_warm_func)

    one_cycle_lr_sched = combine_scheds([0.25, 0.75], [cos_sched.scaled(0.04, 1.), cos_sched.scaled(1., 1e-5)])
    fastai1_lr_sched = combine_scheds([0.3, 0.7], [cos_sched.scaled(0.04, 1.), cos_sched.scaled(1., 4e-6)])
    mod_one_cycle_lr_sched = combine_scheds([0.25, 0.75], [cos_sched.scaled(1e-5, 1.), cos_sched.scaled(1., 1e-5)])

    if not isinstance(sched_type, str):
        base_sched = sched_type
    elif sched_type == 'linear':
        return lin_sched
    elif sched_type == 'constant' or sched_type == 'flat':
        return ConstantSchedule(1.0)
    elif sched_type == 'one_cycle':
        base_sched = one_cycle_lr_sched
    elif sched_type == 'two_cycle':
        base_sched = combine_scheds([0.5, 0.5], [one_cycle_lr_sched] * 2)
    elif sched_type == 'three_cycle':
        base_sched = combine_scheds([0.25, 0.25, 0.5], [one_cycle_lr_sched] * 3)
    elif sched_type == 'four_cycle':
        base_sched = combine_scheds([0.125, 0.125, 0.25, 0.5], [one_cycle_lr_sched] * 4)
    elif sched_type == 'c4':
        base_sched = combine_scheds([0.125, 0.125, 0.25, 0.5], [mod_one_cycle_lr_sched] * 4)
    elif sched_type == 'c5':
        base_sched = combine_scheds([0.0625, 0.0625, 0.125, 0.25, 0.5], [mod_one_cycle_lr_sched] * 5)
    elif sched_type == 'long_plateau':
        base_sched = combine_scheds([0.2, 0.6, 0.2],
                                    [cos_sched.scaled(0.04, 1), ConstantSchedule(1.0), cos_sched.scaled(1, 1e-5)])
    elif sched_type == 'sched1':
        base_sched = connect_cos_scheds([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.04, 1.0, 0.01, 1.0, 1.0, 1e-5])
    elif sched_type == 'sched2':
        base_sched = connect_cos_scheds([0.0, 0.125, 0.375, 0.5, 0.75, 1.0], [0.04, 1.0, 0.05, 1.0, 1.0, 1e-5])
    elif sched_type == 'sched3':
        base_sched = connect_cos_scheds([0.0, 8 / 64, 16 / 64, 24 / 64, 32 / 64, 56 / 64, 1.0],
                                        [1e-3, 1.0, 1.0, 1e-3, 1.0, 1.0, 1e-3])
    elif sched_type == 'sched4':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
    elif sched_type == 'sched5':
        base_sched = connect_cos_scheds([0.0, 0.75, 1.0], [0.04, 1.0, 1e-5])
    elif sched_type == 'sched6':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.5, 0.5], [base_sched] * 2)
    elif sched_type == 'sched7':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.25, 0.25, 0.5], [base_sched] * 3)
    elif sched_type == 'sched8':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.125, 0.125, 0.25, 0.5], [base_sched] * 4)
    elif sched_type == 'sched9':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.125]*8, [base_sched] * 8)
    elif sched_type == 'sched10':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.0625, 0.0625, 0.125, 0.25, 0.5], [base_sched] * 5)
    elif sched_type == 'sched11':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.125, 0.125, 0.25, 0.5],
                                    [ConstantSchedule(lr) * base_sched for lr in [0.6, 0.8, 1.0, 1.5]])
    elif sched_type == 'sched12':
        base_sched = connect_cos_scheds([0.0, 0.5, 1.0], [0.04, 1.0, 1e-5])
        base_sched = combine_scheds([0.125, 0.125, 0.25, 0.5],
                                    [ConstantSchedule(lr) * base_sched for lr in [1.0, 1.0, 1.0, 1.5]])
    elif sched_type == 'custom1':
        sched = connect_cos_scheds([0.0, 0.5, 1.0], [4e-2, 1.0, 1e-5])
        base_sched = combine_scheds([0.5, 0.5], [sched] * 2)
    elif sched_type == 'flat_anneal':
        base_sched = combine_scheds([0.6, 0.4], [ConstantSchedule(1.0), cos_sched.scaled(1., 1e-5)])
    elif sched_type == 'flat_cos':
        base_sched = combine_scheds([0.5, 0.5], [ConstantSchedule(1.0), cos_sched.scaled(1., 0.)])
    elif sched_type == 'cos_anneal':
        base_sched = cos_sched.scaled(1.0, 1e-4)
    elif sched_type == 'fastai1':
        base_sched = fastai1_lr_sched
    elif sched_type == 'cos_warm':
        base_sched = cos_warm_sched
    elif sched_type == 'cos_warm_4':
        base_sched = connect_cos_scheds([0.0, 1/15, 1/15, 3/15, 3/15, 7/15, 7/15, 1.0],
                                        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    elif sched_type == 'datarobot':
        # described in https://www.youtube.com/watch?v=WPQOkoXhdBQ
        base_sched = combine_scheds([0.25, 0.5, 0.25], [cos_sched.scaled(0.1, 1.), cos_sched.scaled(1., 0.1),
                                                                cos_sched.scaled(0.1, 0.003)])
    elif sched_type == 'one_cycle_0.1':
        base_sched = combine_scheds([0.1, 0.9], [cos_sched.scaled(0.04, 1.), cos_sched.scaled(1., 1e-5)])
    elif sched_type == 'one_cycle_mom':
        base_sched = combine_scheds([0.25, 0.75], [cos_sched.scaled(0.95, 0.85), cos_sched.scaled(0.85, 0.95)])
    elif sched_type == '1-1/step':
        base_sched = StepFunctionSchedule(lambda step: 1-1/(step+1))
    elif sched_type == 'epoch_length':
        base_sched = EpochLengthSqMomSchedule()
    elif sched_type == 'epoch_length_2':
        base_sched = EpochLengthSqMomSchedule(base_value=0.1)
    elif sched_type == 'epoch_length_3':
        base_sched = EpochLengthSqMomSchedule(base_value=0.05)
    elif sched_type == 'cos_log_15':
        base_sched = FunctionSchedule(lambda t: 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 15 * t))))
    elif sched_type == 'cos_log_31':
        base_sched = FunctionSchedule(lambda t: 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 31 * t))))
    elif sched_type == 'cos_log_63':
        base_sched = FunctionSchedule(lambda t: 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 63 * t))))
    elif sched_type == 'cos_log_31_sq_mom':
        base_sched = FunctionSchedule(lambda t: np.exp(-0.05 * 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 31 * t))))
                                                - 1e-8)
    elif sched_type == 'cos_sched':
        base_sched = cos_sched.scaled(1., 0.)
    elif sched_type == 'cos':
        base_sched = cos_sched.scaled(1., 0.)
    elif sched_type == 'cos_increasing':
        base_sched = cos_sched.scaled(0., 1.)
    elif sched_type == 'lin_cos_log_15':
        base_sched = FunctionSchedule(lambda t: 2 * t * 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 15 * t))))
    elif sched_type == 'lin2_cos_log_15':
        base_sched = FunctionSchedule(lambda t: (0.5 + t) * 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 15 * t))))
    elif sched_type == 'lin3_cos_log_15':
        base_sched = FunctionSchedule(lambda t: (1.5 - t) * 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + 15 * t))))
    elif isinstance(sched_type, str) and sched_type.startswith('coslog'):
        n_cycles = int(sched_type[len('coslog')])
        base_sched = FunctionSchedule(CoslogFunc(n_cycles))
        # base_sched = FunctionSchedule(lambda t: 0.5 * (1 - np.cos(2 * np.pi * np.log2(1 + (2**n_cycles-1) * t))))
    elif sched_type == 'warmup_0.05_cos':
        base_sched = connect_cos_scheds([0.0, 0.05, 1.0],
                                        [0.0, 1.0, 0.0])

    if base_sched is None:
        raise ValueError(f'Unknown schedule type "{sched_type}"')
    return base_sched



