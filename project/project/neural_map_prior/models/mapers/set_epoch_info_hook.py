# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SetEpochInfoHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(self,
                 start=None,
                 interval=1,
                 by_epoch=True):

        self.start = start
        self.interval = interval
        self.by_epoch = by_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            # model = model.module.head
            model = model.module
        model.set_epoch(epoch)

        map_slice_float_dict_key = dict.fromkeys(model.gm.map_slice_float_dict.keys(), [])
        print('start before_train_epoch removing map_slice_float_dict content !!!!!!!!!!!!!!!')
        for k in map_slice_float_dict_key.keys():
            print(f'remove {k}')
            del model.gm.map_slice_float_dict[k]

        map_slice_int_dict_key = dict.fromkeys(model.gm.map_slice_int_dict.keys(), [])
        print('* 3 start before_train_epoch removing map_slice_int_dict content !!!!!!!!!!!!!!!')
        for k in map_slice_int_dict_key.keys():
            print(f'remove {k}')
            del model.gm.map_slice_int_dict[k]
        print('end before_train_epoch removing map_slice_int_dict content !!!!!!!!!!!!!!!')

    def after_train_epoch(self, runner):
        if not self._should_evaluate(runner):
            return
        model = runner.model
        if is_module_wrapper(model):
            # model = model.module.head
            model = model.module

        map_slice_float_dict_key = dict.fromkeys(model.gm.map_slice_float_dict.keys(), [])
        print('start after_train_epoch removing map_slice_float_dict content !!!!!!!!!!!!!!!')
        for k in map_slice_float_dict_key.keys():
            print(f'remove {k}')
            del model.gm.map_slice_float_dict[k]

        map_slice_int_dict_key = dict.fromkeys(model.gm.map_slice_int_dict.keys(), [])
        print('* 3 start before_train_epoch removing map_slice_int_dict content !!!!!!!!!!!!!!!')
        for k in map_slice_int_dict_key.keys():
            print(f'remove {k}')
            del model.gm.map_slice_int_dict[k]
        print('end before_train_epoch removing map_slice_int_dict content !!!!!!!!!!!!!!!')

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True
