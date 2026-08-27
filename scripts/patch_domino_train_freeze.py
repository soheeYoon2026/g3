"""Add config-driven submodule freezing to the vendored DoMINO train.py.

The serving model is fine-tuned with the pretrained geometry encoder detached, so
comparing the official pipeline against it fairly needs the same option here.
Idempotent: re-running leaves an already-patched file untouched.
"""

import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

MARKER = "# AOX: optional submodule freezing"
if MARKER in text:
    print("이미 패치됨")
    raise SystemExit(0)

anchor = '    logger.info(f"Model summary:\\n{torchinfo.summary(model, verbose=0, depth=2)}\\n")\n'
if anchor not in text:
    raise SystemExit("anchor not found — train.py layout changed")

block = anchor + f"""
    {MARKER}. With only tens of fine-tuning cases, training all 372 tensors
    # destroys the DrivAerML geometry representation; freezing it reproduces the
    # serving model's setup inside the official pipeline.
    _frozen = list(getattr(cfg.train, "freeze_modules", None) or [])
    if _frozen:
        for _name in _frozen:
            _module = getattr(model, _name, None)
            if _module is None:
                raise ValueError(f"freeze_modules: no submodule '{{_name}}' on DoMINO")
            for _p in _module.parameters():
                _p.requires_grad_(False)
        _total = sum(p.numel() for p in model.parameters())
        _train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Froze {{_frozen}}: {{_train:,}}/{{_total:,}} parameters trainable")
"""
text = text.replace(anchor, block, 1)

old_opt = """    optimizer = optimizer_class(
        model.parameters(),"""
new_opt = """    optimizer = optimizer_class(
        [p for p in model.parameters() if p.requires_grad],"""
if old_opt not in text:
    raise SystemExit("optimizer construction not found — train.py layout changed")
text = text.replace(old_opt, new_opt, 1)

path.write_text(text)
print("패치 완료:", path)
