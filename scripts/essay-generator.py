import openai
import time
import itertools
import uuid
import csv

# Set up the OpenAI API client
openai.api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

tones = ['formal', 'informal', 'persuasive', 'optimistic',
         'friendly', 'analytical', 'assertive', 'encouraging', 'narrative', 'expository']
ageGroups = ['child', 'teenager']

# Bias combinations
combinations = list(itertools.product(tones, ageGroups))

# Essay topics
topics = [
    (1, 'Write a letter to your local newspaper in which you state your opinion on the effects computers have on people'),
    (2, 'Write to a newspaper reflecting your vies on censorship in libraries and if certain materials should be removed from the shelves if they are found offensive'),
    (3, 'Write about what factors can affect a lost cyclist trying to get to the nearest town'),
    (6, 'Write about the obstacles the builders of the Empire State Building faced in attempting to allow dirigibles to dock there'),
    (7, 'Write about patience. Being patient means that you are understanding and tolerant. A patient person experience difficulties without complaining'),
    (8, 'We all understand the benefits of laughter. Tell a true story in which laughter was one element or part of it'),
]

# Set up the text prompt
prompt = 'Generate a 200 word max essay about:'

# CSV file headers
headers = ['essay_id', 'essay_set', 'essay']


def generatePrompt(topic, combination):
    return f'{prompt} {topic}. Use a {combination[0]} tone and make it sound like it was written by a {combination[1]}'


def generateEssay(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Specify the language model to use
            prompt=prompt,
            max_tokens=500,  # Set the maximum number of tokens in the response
            n=1,  # Set the number of responses to generate
            stop=None,  # Set the stopping sequence for the response generation
        )
        # Extract the generated text from the API response
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        if e.http_status == 429:
            print("API rate limit exceeded. Waiting for 5 seconds...")
            time.sleep(5)
            return generateEssay(prompt)
        else:
            print("OpenAI API error: {}".format(e))
            return generateEssay(prompt)
    except Exception as e:
        print("Unexpected error: {}".format(e))


def writeCSV(writer, set, essay):
    formattedEssay = " ".join(essay.replace(
        '\n', ' ').replace('\r', '').split())
    writer.writerow([uuid.uuid4(), set, formattedEssay])


def main():
    NUMBER_OF_ITERATIONS = 10
    fileName = uuid.uuid4()
    with open(f'{fileName}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row to the CSV file
        writer.writerow(headers)

        for topic in topics:
            for combination in combinations:
                for i in range(NUMBER_OF_ITERATIONS):
                    newPrompt = generatePrompt(topic[1], combination)
                    response = generateEssay(newPrompt)
                    writeCSV(writer, topic[0], response)


if __name__ == "__main__":
    main()
