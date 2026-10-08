import os
import torch


# from pypots.imputation import iTransformer

from models import Transformer, PatchTST, Vanilla_Transformer_MSM, TED, TED_modular
from models import Patch_NTP, Transformer_NTP
from models import Patch_Masked, Patch_NTP_TED
import models.Patch_Masked_Cls as Patch_Masked_Cls


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer':Transformer,
            'PatchTST': PatchTST,
            'Vanilla_Transformer_MSM':Vanilla_Transformer_MSM,
            'Patch_NTP':Patch_NTP,
            'Transformer_NTP':Transformer_NTP,
            'TED':TED,
            'TED_modular': TED_modular,
            'Patch_Masked': Patch_Masked,
            'Patch_NTP_TED': Patch_NTP_TED,
            'Patch_Masked_Cls': Patch_Masked_Cls,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                if getattr(self.args, 'local_rank', 0) == 0:
                    print('Use MPS')
                return device
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            if getattr(self.args, 'local_rank', 0) == 0:
                if self.args.use_multi_gpu:
                    print('Use GPU: cuda{}'.format(self.args.device_ids))
                else:
                    print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            if getattr(self.args, 'local_rank', 0) == 0:
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
