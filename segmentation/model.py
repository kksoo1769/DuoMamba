"""Register DuoMamba with mmseg; import only this task's framework."""
import sys
from pathlib import Path

from mmengine.model import BaseModule
from mmseg.registry import MODELS

# The shared classification implementation lives beside this task directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification.models.duomamba import BackboneDuoMamba


@MODELS.register_module(name='MM_DuoMamba')
class MM_DuoMamba(BaseModule, BackboneDuoMamba):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        BackboneDuoMamba.__init__(self, *args, **kwargs)
