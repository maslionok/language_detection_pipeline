# Import necessary Python modules from the Transformers library
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
# Define the model name to be used for token classification, we use the Impresso NER
# that can be found at "https://huggingface.co/impresso-project/ner-stacked-bert-multilingual"
MODEL_NAME = "emanuelaboros/lang-detect"

lang_pipeline = pipeline("lang-ident", model=MODEL_NAME,
                        trust_remote_code=True,
                        device='cpu')

sentence = "En l'an 1348, au plus fort des ravages de la peste noire à travers l'Europe, le Royaume de France se trouvait à la fois au bord du désespoir et face à une opportunité. À la cour du roi Philippe VI, les murs du Louvre étaient animés par les rapports sombres venus de Paris et des villes environnantes. La peste ne montrait aucun signe de répit, et le chancelier Guillaume de Nogaret, le conseiller le plus fidèle du roi, portait le lourd fardeau de gérer la survie du royaume."

entities = lang_pipeline(sentence)
print(entities)

