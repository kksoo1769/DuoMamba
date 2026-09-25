# Third-party notices

A project-wide license for the original DuoMamba contributions has not yet been selected. Existing third-party notices and licenses continue to apply to their respective portions.

| Source | Relevant portions | License copy |
|---|---|---|
| [Swin Transformer](https://github.com/microsoft/Swin-Transformer) | Classification data, configuration, training, and utility code; headers credit Microsoft and Ze Liu | [MIT](LICENSES/Swin-Transformer-MIT.txt) |
| [MLLA](https://github.com/LeapLabTHU/MLLA) | Vision modules in `classification/models/mamba_util.py`; header credits Dongchen Han; MESA training approach | [MIT](LICENSES/MLLA-MIT.txt) |
| [Mamba](https://github.com/state-spaces/mamba/tree/v2.0.4) | SSM dependency and adaptations in `duomamba.py` and `mhsa.py`; headers credit Tri Dao and Albert Gu | [Apache-2.0](LICENSES/Mamba-Apache-2.0.txt) |
| [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v3.3.0) | Detection base/Swin configurations and adapted training/evaluation tools | [Apache-2.0](LICENSES/MMDetection-Apache-2.0.txt) |
| [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/v1.2.2) | Segmentation base/Swin configurations and adapted training/evaluation tools | [Apache-2.0](LICENSES/MMSegmentation-Apache-2.0.txt) |

Modified entry points add DuoMamba registration, dataset-root overrides, and distributed launch handling. The retained ADE20K configuration uses the paper's multi-scale evaluation settings. Model adaptations and task configurations are described in [release notes](docs/RELEASE_NOTES.md).

License texts were retrieved from the linked upstream projects during release preparation. The version links identify the selected dependency and license sources; exact upstream commit IDs for adapted files are unavailable. Original attribution headers are preserved.
