"""Hugging Face Trainer / TRL integration: one line to log training metrics.

    trainer = SFTTrainer(..., callbacks=[rdp_callback(run)])
"""

from __future__ import annotations

from rdp.metrics import Run


def rdp_callback(run: Run):
    from transformers import TrainerCallback  # optional dependency: pip install rdp[hf]

    class RDPCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            numeric = {k: v for k, v in (logs or {}).items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if numeric:
                run.log(numeric, step=state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            run.flush()

    return RDPCallback()
