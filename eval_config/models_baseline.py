from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr="qwen35_2b_original",
        path="model_assets/qwen3.5-2b",
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
]
