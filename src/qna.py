import torch
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.tokenization import Tokenizer
from farm.infer import Inferencer
from pprint import pprint
import numpy as np
from sklearn.preprocessing import normalize
# from IPython.display import clear_output

def filter_answer(answer=[]):
    answer_prob = []
    total = 0
    for ans in answer:
        if  not ans['answer']=="no_answer":
            answer_prob.append((ans['answer'], ans['score']))
            total = total + ans['score']
    answer_prob = [(answer[0], round(answer[1]/total, 2)) for answer in answer_prob]
    return answer_prob


def load_qna_model(model_dir):
    if torch.cuda.is_available():
        nlp = Inferencer.load(model_dir, task_type="question_answering", gpu=True)
        nlp.save(model_dir)
    else:
        nlp = Inferencer.load(model_dir, task_type="question_answering", gpu=False)
        nlp.save(model_dir)
    return nlp
    

def answer_ques(context, question, nlp):
    if isinstance(question, str):
        question = [question]
    qa_input = [{"questions": question,
                "text": context}]
    res = nlp.inference_from_dicts(dicts=qa_input)
    result = filter_answer(res[0]['predictions'][0]['answers'])
    return res

# answer_ques(
#     context= """Nepal, the landlocked multiethnic, multilingual, multi-religious country, is situated north of India in the Himalayas, in the region where, about 40 to 50 million years ago, the Indian subcontinent has crashed into Asia. Because of that accident, Nepal has some of the world's highest mountains including Sagarmatha (Mt. Everest, 8848m, which it shares with Tibet (by now a province of China). The highest mountain on Earth is towering above populated valleys and forested plains.
# Somewhere here in the Kapilavastu district, there is a place called Lumbini where in about 500 B.C.E. Queen Mayadevi is said to have given birth to Siddhartha Gautama, better known as Buddha.

# Nepal can be divided broadly into three ecological zones: the lowland, the midland and the highland.
# The altitude of the Himalayan Region (the highland) ranges between 4877 m - 8848 m, It includes 8 of the highest 14 summits in the world, which exceed altitude of 8000 meters including Mount Everest.

# The mountain region accounts for about 64 percent of total land area, which is formed by the Mahabharat range that soars up to 4877 m and the lower Churia range.
# The lowland Terai, the flat river plain of the Ganges with a belt of marshy grasslands, savannas, and forests, occupies about 17 percent of the total land area of the country. """,
# question= "how old is lumbini?"
# )