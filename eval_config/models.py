from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

_model_configs = [
    ("qwen35_2b_ttt_on", "data/output/qwen35_2b_ttt/hf_step5000_ttt_on"),
    ("qwen35_2b_ttt_off", "data/output/qwen35_2b_ttt/hf_step5000_ttt_off"),
]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=name,
        path=path,
        model_kwargs=dict(
            torch_dtype="bfloat16",
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=False,
            temperature=0.0,
            top_p=0.95,
            top_k=20,
        ),
        max_out_len=64,
        batch_size=1,
        max_seq_len=131072,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
    for name, path in _model_configs
]
