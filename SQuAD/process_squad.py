import pandas as pd
def process_squad(file):
    data = pd.read_json(file)
    contexts = []
    questions = []
    answers_text = []
    answers_start = []
    for i in range(data.shape[0]):
        topic = data.iloc[i,0]['paragraphs']
        for sub_para in topic:
            for q_a in sub_para['qas']:
                questions.append(q_a['question'])
                answers_start.append(q_a['answers'][0]['answer_start'])
                answers_text.append(q_a['answers'][0]['text'])
                contexts.append(sub_para['context'])
    df = pd.DataFrame({"context":contexts, "question": questions, "answer_start": answers_start, "text": answers_text})
    return df

# if __name__=='__main__':
#     train_data = process_squad("SQuAD_data/train-v1.1.json")
#     valid_data = process_squad("SQuAD_data/dev-v1.1.json")
#     print(train_data.head(5))
