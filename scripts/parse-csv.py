import csv
import json
import random

essay_sets = [1, 2, 3, 6, 7, 8]
topics = [
    (1, 'Write a letter to your local newspaper in which you state your opinion on the effects computers have on people'),
    (2, 'Write to a newspaper reflecting your vies on censorship in libraries and if certain materials should be removed from the shelves if they are found offensive'),
    (3, 'Write about what factors can affect a lost cyclist trying to get to the nearest town'),
    (6, 'Write about the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there'),
    (7, 'Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining'),
    (8, 'We all understand the benefits of laughter. Tell a true story in which laughter was one element or part of it'),
]
topic_dict = {topic[0]: topic[1] for topic in topics}
dictionaries = []  # [{question: "", human_answers: [], chatgp_answers: []}]


def getDict(set):
    for dictionary in dictionaries:
        question = topic_dict[set]
        if dictionary['question'] == question:
            return dictionary
    return None


def parseChatGPT():
    # Open the CSV file
    with open('dataset-chatgpt.csv', 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        # Split the CSV rows into two lists
        for row in reader:
            set = int(row['essay_set'])
            essay = row['essay']
            dictionary = getDict(set)
            if dictionary is not None:
                dictionary['chatgp_answers'].append(essay)
            else:
                dictionaries.append(
                    {'question': topic_dict[set], 'human_answers': [], 'chatgp_answers': [essay]})


def parseHuman():
    # Open the CSV file
    with open('dataset-human.csv', 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        # Split the CSV rows into two lists
        for row in reader:
            set = int(row['essay_set'])
            essay = row['essay']
            if set in [4, 5]:  # Does NOT include essay sets 4 and 5
                continue
            dictionary = getDict(set)
            if dictionary is not None:
                dictionary['human_answers'].append(essay)
            else:
                dictionaries.append(
                    {'question': topic_dict[set], 'human_answers': [essay], 'chatgp_answers': []})


def trainAndTest():
    # 80% train, 20% test
    train = []
    test = []
    for dictionary in dictionaries:
        human_answers = dictionary['human_answers']
        chatgp_answers = dictionary['chatgp_answers']
        # Shuffle the answers
        random.shuffle(human_answers)
        random.shuffle(chatgp_answers)

        train.append({'question': dictionary['question'], 'human_answers': human_answers[:int(len(
            human_answers) * 0.8)], 'chatgp_answers': chatgp_answers[:int(len(chatgp_answers) * 0.8)]})
        test.append({'question': dictionary['question'], 'human_answers': human_answers[int(len(
            human_answers) * 0.8):], 'chatgp_answers': chatgp_answers[int(len(chatgp_answers) * 0.8):]})
    return train, test


def convertJSONLformat(objectList):
    newList = []
    for obj in objectList:
        question = obj['question']
        human_answers = obj['human_answers']
        for human_answer in human_answers:
            newList.append(
                {'question': question, 'answer': human_answer, 'chatgpt': False})
        chatgp_answers = obj['chatgp_answers']
        for chatgp_answer in chatgp_answers:
            newList.append(
                {'question': question, 'answer': chatgp_answer, 'chatgpt': True})
    return newList


def saveJSONL(objectList, filename):
    with open(filename, 'w') as f:
        for obj in objectList:
            f.write(json.dumps(obj) + '\n')


if __name__ == '__main__':
    parseHuman()
    parseChatGPT()
    train, test = trainAndTest()

    # Save the JSON object to a file
    saveJSONL(convertJSONLformat(train), 'train.jsonl')
    saveJSONL(convertJSONLformat(test), 'test.jsonl')
