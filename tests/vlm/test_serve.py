from config.settings import load_profile
from docparse.vlm.serve import compose_env


def test_compose_env_fast_and_accurate() -> None:
    p = load_profile("consumer24")
    fast = compose_env(p, "fast")
    assert fast["MODEL_ID"] == "lightonai/LightOnOCR-3-0.8B"
    assert fast["MAX_NUM_SEQS"] == "256" and fast["VLLM_CPUSET"] == "0-3"
    assert compose_env(p, "accurate")["MODEL_ID"].endswith("4B")
