from chatterbot import ChatBot # Chatbot library
from chatterbot.trainers import ListTrainer # Training function
import time
import nltk # NLP processing toolkit
import re #Handling regular expressions
from pathlib import Path

time.clock = time.time
nltk.download('punkt_tab') # Enable NLP Processing, used by ChatBot


# tokenizes training data based on Q/A so it's digestible by ML model
def cleanData(textData):
    p = Path(__file__).with_name(textData)
    questions = []
    answers = []
    answerPart = ""
    with p.open('r', encoding="utf-8") as file:
        for line in file.readlines():
            text = re.sub("\n" ,"", line)
            if '?' in text:
                questions.append(text)
                answers.append(answerPart)
            else:
                answerPart += text
            
    return questions, answers

# Trains the chatbot on the set of questions and answers
def trainChatbot(chatbot):
    questions, answers = cleanData("MABenefitsQA.txt")
    Trainer = ListTrainer(chatbot)
    for i in range(min(len(questions), len(answers))):
        pair = (questions[i], answers[i])
        Trainer.train(pair)
        
    return chatbot


def main():
    chatbot = ChatBot("Chatpot")
    trainedChatbot = trainChatbot(chatbot) # initializing and training chatbot

    exitConditions = ("quit", "exit")

    while True:
        query = input(">")
        if query in exitConditions:
            break
        else:
            print(f"Response Message: {trainedChatbot.get_response(query)}")

if __name__ == "__main__":
    main()