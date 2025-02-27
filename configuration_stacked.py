from transformers import PretrainedConfig
import os


class ImpressoConfig(PretrainedConfig):
    model_type = "floret"

    def __init__(self, filename="LID-40-3-2000000-1-4.bin", **kwargs):
        super().__init__(**kwargs)
        self.filename = filename

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Bypass JSON loading and create config directly
        print(f"Loading ImpressoConfig from {pretrained_model_name_or_path}")
        print(os.getcwd())
        config = cls(filename="LID-40-3-2000000-1-4.bin", **kwargs)
        return config


# Register the configuration with the transformers library
ImpressoConfig.register_for_auto_class()

# Register the custom pipeline
# PIPELINE_REGISTRY.register_pipeline(
#     task="lang-ident",
#     pipeline_class=LangIdentPipeline,
#     model=AutoModelForSequenceClassification,
#     tokenizer=AutoTokenizer,
# )
#
# print("Custom pipeline 'lang-ident' registered successfully.")
