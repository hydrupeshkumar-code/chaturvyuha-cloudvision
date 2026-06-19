from ai.fusion.pipeline import FusionPipeline

pipeline = None

MODEL_PATH = "models/generator_best_lissiv.pth"


def get_pipeline():
    global pipeline

    if pipeline is None:
        pipeline = FusionPipeline(MODEL_PATH)

    return pipeline