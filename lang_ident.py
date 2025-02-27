from transformers import Pipeline


class LangIdentPipeline(Pipeline):

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "text" in kwargs:
            preprocess_kwargs["text"] = kwargs["text"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, **kwargs):
        print("this is preprocessing:")
        print(text)
        return text

    def _forward(self, text):
        # Extract label and confidence
        predictions, probabilities = self.model.predict([text], k=1)

        label = predictions[0][0].replace("__label__", "")  # Remove __label__ prefix
        confidence = float(
            probabilities[0][0]
        )  # Convert to float for JSON serialization

        # Format as JSON-compatible dictionary
        model_output = {"label": label, "confidence": round(confidence * 100, 2)}

        print("Formatted Model Output:", model_output)
        return model_output

    def postprocess(self, outputs, **kwargs):
        return outputs


# PIPELINE_REGISTRY.register_pipeline(
#     task="language-detection",
#     pipeline_class=Pipeline_One,
#     default={"model": None},
# )
