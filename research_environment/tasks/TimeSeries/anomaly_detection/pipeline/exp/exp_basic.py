import os
import torch
import importlib  # <--- 必须添加这个库

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {}

        # 1. 定义支持的模型名称列表
        # 这里列出了你要求的所有模型，仅作为名称索引，不会触发导入
        supported_models = [
            'TimesNet',
            'Autoformer',
            'Transformer',
            'Nonstationary_Transformer',
            'DLinear',
            'TimeEmb',
            'FEDformer',
            'Informer',
            'LightTS',
            'Reformer',
            'ETSformer',
            'PatchTST',
            'Pyraformer',
            'MICN',
            'Crossformer',
            'FiLM',
            'iTransformer',
            'Koopa',
            'TiDE',
            'FreTS',
            'MambaSimple',
            'TimeMixer',
            'TSMixer',
            'SegRNN',
            'TemporalFusionTransformer',
            "SCINet",
            'PAttn',
            'TimeXer',
            'WPMixer',
            'MultiPatchFormer',
            'KANAD',
            'MSGNet',
            'TimeFilter',
            'Sundial',
            'TimeMoE',
            'Chronos',
            'Moirai',
            'TiRex',
            'TimesFM',
            'Toto',
            'Chronos2',
            'PSWI',
            'CATCH',
            'P_sLSTM',
            'MtsCID',
            'CrossAD',
            'InterpGN',
            'SGN',
        ]

        if self.args.model in supported_models:
            try:
                # 关键修改：直接导入 models 包下的具体模块文件
                # 例如：import models.TimesNet
                module_name = f'models.{self.args.model}' 
                mod = importlib.import_module(module_name)
                
                # 将导入的模块（文件）存入字典
                # 这样后面 _build_model 里的 mod.Model(args) 才能正常工作
                self.model_dict[self.args.model] = mod
                
            except ImportError as e:
                print(f"Error: Failed to import model module '{module_name}'.")
                print(f"Detailed error: {e}")
                # 如果是 TimesNet 报错，说明 models/TimesNet.py 可能不存在或有语法错误
                raise e

        # 3. 特殊处理 Mamba (保留原有逻辑)
        if self.args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            try:
                from models import Mamba
                self.model_dict['Mamba'] = Mamba
            except Exception as e:
                print(f"Error importing Mamba: {e}")

        # 4. 初始化设备和模型
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        # 此时 self.model_dict 里应该只有刚才动态导入的那一个模型
        if self.args.model in self.model_dict:
            return self.model_dict[self.args.model].Model(self.args)
        else:
            raise NotImplementedError(f"Model {self.args.model} is not implemented or failed to import.")

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # 检查外层调度器是否已经设置了 CUDA_VISIBLE_DEVICES
            # 如果已设置（例如通过 Singularity 容器或 replicate.py），则不要覆盖
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            
            if cuda_visible_devices is not None:
                # 外层已经设置了 GPU 映射，使用 cuda:0（映射后的第一个可见 GPU）
                device = torch.device('cuda:0')
                print(f'Use GPU: cuda:0 (mapped from physical GPU via CUDA_VISIBLE_DEVICES={cuda_visible_devices})')
            else:
                # 没有外层设置，使用传统方式
                os.environ["CUDA_VISIBLE_DEVICES"] = str(
                    self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
# import os
# import torch
# from models import Autoformer, Chronos, Chronos2, TimesNet, Transformer, Nonstationary_Transformer, DLinear, FEDformer, Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, WPMixer, MultiPatchFormer, KANAD, MSGNet, TimeFilter, Sundial, TimeMoE, Moirai, TiRex, TimesFM, Toto


# class Exp_Basic(object):
#     def __init__(self, args):
#         self.args = args
#         self.model_dict = {
#             'TimesNet': TimesNet,
#             'Autoformer': Autoformer,
#             'Transformer': Transformer,
#             'Nonstationary_Transformer': Nonstationary_Transformer,
#             'DLinear': DLinear,
#             'FEDformer': FEDformer,
#             'Informer': Informer,
#             'LightTS': LightTS,
#             'Reformer': Reformer,
#             'ETSformer': ETSformer,
#             'PatchTST': PatchTST,
#             'Pyraformer': Pyraformer,
#             'MICN': MICN,
#             'Crossformer': Crossformer,
#             'FiLM': FiLM,
#             'iTransformer': iTransformer,
#             'Koopa': Koopa,
#             'TiDE': TiDE,
#             'FreTS': FreTS,
#             'MambaSimple': MambaSimple,
#             'TimeMixer': TimeMixer,
#             'TSMixer': TSMixer,
#             'SegRNN': SegRNN,
#             'TemporalFusionTransformer': TemporalFusionTransformer,
#             "SCINet": SCINet,
#             'PAttn': PAttn,
#             'TimeXer': TimeXer,
#             'WPMixer': WPMixer,
#             'MultiPatchFormer': MultiPatchFormer,
#             'KANAD': KANAD,
#             'MSGNet': MSGNet,
#             'TimeFilter': TimeFilter,
#             'Sundial': Sundial,
#             'TimeMoE': TimeMoE,
#             'Chronos': Chronos,
#             'Moirai': Moirai,
#             'TiRex': TiRex,
#             'TimesFM': TimesFM,
#             'Toto': Toto,
#             'Chronos2': Chronos2
#         }
#         if args.model == 'Mamba':
#             print('Please make sure you have successfully installed mamba_ssm')
#             from models import Mamba
#             self.model_dict['Mamba'] = Mamba

#         self.device = self._acquire_device()
#         self.model = self._build_model().to(self.device)

#     def _build_model(self):
#         raise NotImplementedError
#         return None

#     def _acquire_device(self):
#         if self.args.use_gpu and self.args.gpu_type == 'cuda':
#             os.environ["CUDA_VISIBLE_DEVICES"] = str(
#                 self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
#             device = torch.device('cuda:{}'.format(self.args.gpu))
#             print('Use GPU: cuda:{}'.format(self.args.gpu))
#         elif self.args.use_gpu and self.args.gpu_type == 'mps':
#             device = torch.device('mps')
#             print('Use GPU: mps')
#         else:
#             device = torch.device('cpu')
#             print('Use CPU')
#         return device

#     def _get_data(self):
#         pass

#     def vali(self):
#         pass

#     def train(self):
#         pass

#     def test(self):
#         pass