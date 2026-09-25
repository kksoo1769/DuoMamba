"""Validate release configuration wiring without importing CUDA or task frameworks."""
import ast
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "classification"))
from config import get_config


def check_classification_config():
    """Exercise configuration inheritance and CLI precedence through the real loader."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        (base / 'base.yaml').write_text(
            'MODEL:\n  TYPE: duomamba\n  DUOMAMBA:\n    EMBED_DIM: 96\n'
            '    CHUNK_SIZE: [32, 32, 64, 64]\n'
            '    MIXER_TYPES: [DuoScanMixer, DuoScanMixer, DuoScanMixer, MHSA]\n')
        (base / 'child.yaml').write_text(
            'BASE: [base.yaml]\nMODEL:\n  DUOMAMBA:\n    D_STATE: 32\n')
        (base / 'scalar.yaml').write_text(
            'BASE: [child.yaml]\nMODEL:\n  DUOMAMBA:\n    CHUNK_SIZE: 64\n')
        cfg = get_config(SimpleNamespace(cfg=str(base / 'child.yaml'), opts=None))
        assert cfg.MODEL.TYPE == 'duomamba'
        assert cfg.MODEL.DUOMAMBA.EMBED_DIM == 96
        assert cfg.MODEL.DUOMAMBA.D_STATE == 32
        assert cfg.MODEL.DUOMAMBA.MIXER_TYPES == ['DuoScanMixer'] * 3 + ['MHSA']
        assert cfg.MODEL.DUOMAMBA.CHUNK_SIZE == [32, 32, 64, 64]
        mixed = get_config(SimpleNamespace(cfg=str(base / 'scalar.yaml'), opts=[
            'MODEL.DUOMAMBA.CHUNK_SIZE', '[16, 32, 64, 64]',
            'MODEL.DUOMAMBA.CHUNK_SIZE', '128',
            'MODEL.DUOMAMBA.D_STATE', '8', 'MODEL.DUOMAMBA.D_STATE', '16']))
        assert mixed.MODEL.DUOMAMBA.CHUNK_SIZE == 128
        assert mixed.MODEL.DUOMAMBA.D_STATE == 16
        assert mixed.MODEL.TYPE == 'duomamba'

        # Run the actual factory with a recording constructor, without importing CUDA.
        factory_path = ROOT / 'classification/models/__init__.py'
        factory = next(node for node in ast.parse(factory_path.read_text()).body
                       if isinstance(node, ast.FunctionDef) and node.name == 'build_duomamba_model')
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        namespace = {'DuoMamba': lambda **kwargs: kwargs}
        exec(compile(module, str(factory_path), 'exec'), namespace)
        kwargs = namespace['build_duomamba_model'](cfg)
        assert kwargs['embed_dim'] == 96 and kwargs['d_state'] == 32
        assert kwargs['chunk_size'] == [32, 32, 64, 64]


def check_cli_overrides(task, action, path):
    """Exercise the actual entry point up to runner setup, without loading CUDA."""
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    # Stop immediately before work-directory/runner setup; run the config-loading
    # statements themselves so regressions in either train or test are caught.
    cutoff = next(i for i, n in enumerate(main.body)
                  if isinstance(n, ast.If) and "args.work_dir" in ast.unparse(n.test))
    main.body = main.body[:cutoff] + [ast.Return(value=ast.Name(id="cfg", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
    config_path = next((ROOT / task / "configs/duomamba").glob("*tiny.py"))
    args = SimpleNamespace(
        config=str(config_path), launcher="none", data_path="/dataset with spaces",
        cfg_options={"model.backbone.pretrained": "/weights/model.pth",
                     "train_dataloader.batch_size": 3,
                     "test_dataloader.dataset.data_root": "/explicit/test"})
    namespace = {"parse_args": lambda: args, "Config": Config,
                 "osp": __import__("os").path,
                 "setup_cache_size_limit_of_dynamo": lambda: None}
    exec(compile(module, str(path), "exec"), namespace)
    cfg = namespace["main"]()
    assert cfg.model.backbone.pretrained == "/weights/model.pth", (task, action)
    assert cfg.train_dataloader.batch_size == 3, (task, action)
    assert cfg.val_dataloader.dataset.data_root == "/dataset with spaces", (task, action)
    assert cfg.test_dataloader.dataset.data_root == "/explicit/test", (task, action)
    if task == "detection":
        for key in ("val_evaluator", "test_evaluator"):
            assert cfg[key].ann_file == "/dataset with spaces/annotations/instances_val2017.json"


def main():
    python_files = list(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(), filename=str(path))
    check_classification_config()

    for size, depth, width, batch in (
        ("tiny", [2, 4, 10, 5], 64, 128),
        ("small", [2, 5, 20, 6], 64, 256),
        ("base", [2, 5, 20, 6], 96, 256),
    ):
        cfg = get_config(SimpleNamespace(
            cfg=str(ROOT / f"classification/configs/duomamba/{size}.yaml"), opts=None))
        assert cfg.MODEL.DUOMAMBA.DEPTHS == depth, size
        assert cfg.MODEL.DUOMAMBA.EMBED_DIM == width, size
        assert cfg.DATA.BATCH_SIZE == batch and cfg.AMP_DTYPE == "bf16", size
        assert cfg.MODEL.TYPE == "duomamba", size
        mixers = cfg.MODEL.DUOMAMBA.MIXER_TYPES
        assert mixers[:2] == ["DuoScanMixer", "DuoScanMixer"] and mixers[3] == "MHSA", size
        assert len(mixers[2]) == depth[2], size
        assert all(m == ("MHSA" if (i + 1) % 5 == 0 else "DuoScanMixer")
                   for i, m in enumerate(mixers[2])), size

    config_count = 0
    for task in ("detection", "segmentation"):
        for path in (ROOT / task / "configs").rglob("*.py"):
            cfg = Config.fromfile(str(path))
            config_count += 1
            if path.parent.name != "duomamba":
                continue
            backbone = cfg.model.backbone
            assert backbone.type == "MM_DuoMamba", path
            assert backbone.pretrained is None, path
            assert "init_cfg" not in backbone and "embed_dims" not in backbone, path
            assert backbone.chunk_size == 64 and backbone.ngroups == 2, path
            assert backbone.mixer_types[:2] == ['DuoScanMixer', 'DuoScanMixer'], path
            assert backbone.mixer_types[3] == 'MHSA', path
            assert all(m == ('MHSA' if (i + 1) % 5 == 0 else 'DuoScanMixer')
                       for i, m in enumerate(backbone.mixer_types[2])), path
            channels = [backbone.embed_dim * 2**i for i in range(4)]
            if task == "detection":
                assert cfg.model.neck.in_channels == channels, path
                expected_epochs = 36 if "_3x" in path.stem else 12
                assert cfg.train_cfg.max_epochs == expected_epochs, path
                assert cfg.param_scheduler[0].end == (1000 if expected_epochs == 36 else 500)
            else:
                assert cfg.model.decode_head.in_channels == channels, path
                assert cfg.model.auxiliary_head.in_channels == channels[2], path
                assert cfg.train_cfg.max_iters == 160000, path
                assert cfg.train_dataloader.batch_size == 2, path
                assert cfg.img_ratios == [0.75, 1.0, 1.25], path
        for action in ("train", "test"):
            check_cli_overrides(task, action, ROOT / task / f"tools/{action}.py")
    print(f"PASS: {len(python_files)} Python sources, 3 classification configs, "
          f"{config_count} OpenMMLab configs, 4 dataset/CLI override regressions, "
          "and classification inheritance/CLI/factory checks.")


if __name__ == "__main__":
    main()
