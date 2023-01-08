from torch.optim.lr_scheduler import _LRScheduler
import types
import warnings
import sys
import math

'''
https://github.com/KyungBong-Ryu/workspace
'''
# Poly Scheduler
class PolyLR(_LRScheduler):
    # modified version of LambdaLR from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    def __init__(self, optimizer, last_epoch=-1, verbose=False, power=0.9, max_epoch=5000):
        self.optimizer = optimizer
        lr_lambda = lambda x : (1 - x/max_epoch)**power
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(PolyLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict


    def load_state_dict(self, state_dict):
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        out_lr_raw = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        out_lr = out_lr_raw[0]
        
        try:    # complex 클래스 발생 문제 해결목적
            if out_lr < 0.0:
                warnings.warn("(PolyLR) lr fixed to 0.0")
                out_lr = 0.0
        except:
            warnings.warn("(PolyLR) lr generated as complex value")
            warnings.warn("(PolyLR) lr fixed to 0.0")
            out_lr = 0.0
        
        return [out_lr]

#=== End of PolyLR


# Poly Scheduler with warm-up

class Poly_Warm_Cos_LR(_LRScheduler):
    # modified version of LambdaLR from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR
    def __init__(self, optimizer, last_epoch=-1, verbose=False
                ,warm_up_steps=500, T_max=50, eta_min=1e-6, style="floor_1"
                ,power=0.9, max_epoch=5000):
        
        list_style_options = ["floor_1", "floor_2", "floor_3", "floor_4"]
        if not style in list_style_options:
            print("(exc) Poly_Warm_Cos_LR -> This style option is not supported")
            sys.exit(-9)
        
        self.optimizer = optimizer
        
        # cosine
        self.warm_up_steps  = warm_up_steps
        self.T_max          = T_max
        if eta_min >= 1:
            print("(exc) Poly_Warm_Cos_LR -> eta_min should be smaller than 1")
            sys.exit(-9)
        self.eta_min        = eta_min
        self.style          = style             
        # min값 처리방식: 최소값 (eta_min) 이하로는 max() 방식으로 처리  -> LR 그래프가 잘린것처럼 형성됨
        #   (1) floor_1:    cosine 범위: -1 ~ 1
        #                   *최초에 생성된 방식
        #   
        #   (2) floor_2:    cosine 범위: 0 ~ 1
        #
        #   (3) floor_3:    cosine 범위: eta_min ~ 1
        
        # poly
        lr_lambda = lambda x : (1 - x/max_epoch)**power
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(Poly_Warm_Cos_LR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict


    def load_state_dict(self, state_dict):
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        out_lr_raw = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        out_lr = out_lr_raw[0]
        
        if self.last_epoch <= self.warm_up_steps:
            if self.style == "floor_1":         # cos 범위: -1 ~ 1
                out_lr = out_lr * math.cos(math.pi * (self.last_epoch / self.T_max))
                
            elif self.style == "floor_2":       # cos 범위: 0 ~ 1
                out_lr = out_lr * ((math.cos(math.pi * (self.last_epoch / self.T_max)) + 1) / 2)
                
            elif self.style == "floor_3":       # cos 범위: eta_min ~ 1
                _cos = ((math.cos(math.pi * (self.last_epoch / self.T_max)) + 1) / 2) * (1 - self.eta_min) + self.eta_min
                out_lr = out_lr * _cos   #-> 상한값이 poly 그래프를 따라 유지되나, 하한값도 poly 그래프를 따라가므로 하한값 잘림 발생
                
            elif self.style == "floor_4":       # cos 범위: eta_min ~ 1
                _cos = (math.cos(math.pi * (self.last_epoch / self.T_max)) + 1) / 2  # 0 ~ 1
                out_lr = out_lr * _cos + self.eta_min * (1 - _cos) # -> 상하한값을 따라 부드럽게 움직이는 형태 -> 잘림현상 없음
            
            else:
                print("(exc) Poly_Warm_Cos_LR -> This style option is not supported")
                sys.exit(-9)
            
            try:    # complex 클래스 발생 문제 해결목적
                if out_lr < self.eta_min:
                    warnings.warn("(Poly_Warm_Cos_LR) lr fixed to eta_min")
                    out_lr = self.eta_min
            except:
                warnings.warn("(Poly_Warm_Cos_LR) lr generated as complex value")
                warnings.warn("(Poly_Warm_Cos_LR) lr fixed to eta_min")
                out_lr = self.eta_min
            
            
            
        else:
            try:    # complex 클래스 발생 문제 해결목적
                if out_lr < 0.0:
                    warnings.warn("(Poly_Warm_Cos_LR) lr fixed to 0.0")
                    out_lr = 0.0
            except:
                warnings.warn("(Poly_Warm_Cos_LR) lr generated as complex value")
                warnings.warn("(Poly_Warm_Cos_LR) lr fixed to 0.0")
                out_lr = 0.0
        
        return [out_lr]
