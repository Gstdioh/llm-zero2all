import torch

import apex
import apex.normalization


rms = apex.normalization.MixedFusedRMSNorm

print(rms)
