from yandex_cloud_ml_sdk import YCloudML
import numpy as np

sdk = YCloudML(
    folder_id="place-your-folder-id-here",
    auth="place-your-authorization-token",
)

model = sdk.models.completions("yandexgpt")

idx = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    30,
    31,
    32,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    34,
    35,
    36,
    37,
    41,
    42,
]

# As we target russian patients we used russain names for tests.
# Any other GPT-like embedding should also work as well.

analytes = [
    "ферритин, нг/мл",
    "витамин В12, пг/мл",
    "фолиевая кислота, нг/мл",
    "аспартатаминотрансфераза, ед/л",
    "аланинаминотрансфераза, ед/л",
    "билирубин прямой, мкмоль/л",
    "билирубин непрямой, мкмоль/л",
    "билирубин общий, мкмоль/л",
    "креатинин, мкмоль/л",
    "мочевина, ммоль/л",
    "белок общий, г/л",
    "лактатдегидрогеназа, ед/л",
    "холестерин, ммоль/л",
    "глюкоза, ммоль/л",
    "мочевая кислота, ммоль/л",
    "альбумин, г/л",
    "гемоглобин, г/л",
    "эритроциты, 10^12/л",
    "средний объем эритроцитов, фл",
    "лейкоциты, 10^9/л",
    "тромбоциты, 10^9/л",
    "нейтрофилы, %",
    "лимфоциты, %",
    "эозинофилы, %",
    "базофилы, %",
    "моноциты, %",
    "С-реактивный белок, мг/л",
    "пол",
    "возраст",
    "средние клетки, %",
    "гранулоциты, %",
    "гликированный гемоглобин, %",
    "ПСА общий, нг/мл",
]

embeds = np.zeros((43, 256))

query_model = sdk.models.text_embeddings("query")
for i, analyte in enumerate(analytes):
    query_embedding = query_model.run(analyte)
    print(query_embedding)

    embeds[idx[i], :] = np.array(query_embedding)

# embeddings.npy is in the repo
np.save("embeddings.npy", embeds)
