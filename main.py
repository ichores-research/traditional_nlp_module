from flair.data import Sentence
from flair.models import SequenceTagger
from helper_functions import * 


# load tagger
tagger = SequenceTagger.load("flair/pos-english")

# make example sentence

raw_sentence = ""

while 1:
    if raw_sentence == "/exit": 
        break

    raw_sentence = str(input("Write the sentence to be analyzed>>> "))
    sentence = Sentence(raw_sentence.lower())
    tagger.predict(sentence)

    tokens = [entity.to_dict()['text'] for entity in sentence if entity.to_dict()['text'] not in banned_tokens]
    labels = [entity.to_dict()['labels'][0]['value'] for entity in sentence if entity.to_dict()['text'] not in banned_tokens]

    separated_sentences = sentence_separator(tokens, labels)



    print("Sentences: ", separated_sentences)


    for s in separated_sentences:

        sentence = Sentence(s.lower())
        # predict NER tags
        tagger.predict(sentence)

        tokens = [entity.to_dict()['text'] for entity in sentence if entity.to_dict()['text'] not in banned_tokens]
        labels = [entity.to_dict()['labels'][0]['value'] for entity in sentence if entity.to_dict()['text'] not in banned_tokens]
        actions = [entity.to_dict()['text'] for entity in sentence if (entity.to_dict()['text'] not in banned_tokens and 'VB' in entity.to_dict()['labels'][0]['value']) ]
        objects = object_extraction(sentence)


        #objects = [entity.to_dict()['text'] for entity in sentence if (entity.to_dict()['text'] not in banned_tokens and 'NN' in entity.to_dict()['labels'][0]['value']) ]

        characteristics = characteristics_extraction(tokens, labels, objects)

        relationships = relationship_extraction(tokens, labels, objects)

        print("Tokens: ", tokens)
        print("Labels: ", labels)
        print("Actions: ", actions)
        print("Objects: ", objects)
        print("Object Characteristics: ", characteristics)
        print("Relationships: ", relationships)
        print("JSON: ", convert_to_JSON(actions, objects, characteristics, relationships))

def scenario_classification(): #TODO: create for calling the LLM-based solution in case the user command is malformed. 
    return 17



#SCENARIOS

# 0-L1	[action1] this; ex: Pick up this object, Pick up this, Pick up that, etc; 
# Patterns: ['VB', 'DT'], ['VB', 'DT', 'NN'] 'NN' is 'object', 'thing', 'stuff';  ['VB', 'PRP']

# 1-L1b	[action1] this [feature1]; ex: Pick up this red thing, Pick up that green object, pick up the smallest object, etc;
# Patterns: ['VB', 'JJ', 'NN'], ['VB', 'DT', 'JJ', 'NN'], ['VB', 'DT', 'JJ', 'NN'], for all, NN needs to be an unkown/non-descriptive reference to an object.

# 2-L1c	[action1] this [feature1] [feature2] [object1]; ex: pick up this red and round object, pick up the long and green object, etc;
# Patterns: ['VB', 'JJ', 'JJ', 'NN' ], ['VB', 'JJ', 'JJ', 'NN'] 'NN' is NOT 'object', 'thing', 'stuff';

# 3-L2a	[action1] an object similar to this: pick up an object similar to this, pick up an object similar to that, pick up something similar to this, etc;
# Patterns: ['VB', 'NN', 'JJ', 'DT'], ['VB', 'NN', 'JJ', 'DT', 'NN'], ['VB', 'NN', 'DT', 'VB', 'IN', 'DT', 'NN'], ['VB', 'NN', 'WDT', 'VB', 'IN', 'DT', 'NN']

# 4-L2b	[action1] an object similar to this in [feature_class1]; ex: pick up an object is similar size to this, pick an object with the same color as this one, etc; 
# Patterns: ['VB', 'NN', 'NN', 'JJ', 'DT', 'NN'], ['VB', 'NN', 'JJ', 'DT', 'IN', 'NN']
# first NN should be non-descriptive, second NN needs to be a feature class, such as size, shape, color, format, width, format, etc. 

# 5-L2c	[action1] the object [position1] to this object; ex: pick up the object to the right of this one, pick up the object in front of that one, etc;
# Patterns: ['VB', 'NN', 'IN', 'DT', 'NN'], ['VB', 'NN', 'RB', 'DT', 'NN'], ['VB', 'NN', 'NN', 'DT', 'NN'], ['VB', 'NN', 'IN', 'NN'], ['VB', 'NN', 'RB', 'NN'], ['VB', 'NN', 'NN', 'NN'], 
#           ['VB', 'NN', 'IN', 'DT'], ['VB', 'NN', 'RB', 'DT'], ['VB', 'NN', 'NN', 'DT'] - 
# Can be summarized by ['VB', 'NN', <whatever>, 'DT', 'NN'], ['VB', 'NN', <whatever>, 'DT'], ['VB', 'NN', <whatever>, 'NN']

# 6-L2d	[action1] the [feature1] object [position1] to this object; ex: pick up the green object left of this one; pick up the big object in front of this object, etc; 
# Patterns: ['VB', 'JJ', 'NN', <whatever>, 'DT', 'NN'], Patterns: ['VB', 'JJ', 'NN', <whatever>, 'NN']

# 7-L3	[action1] it here; ex: move this object here, move that object there, move that here, etc;
# Patterns: ['VB', 'DT', 'RB'], ['VB', 'PRP', 'RB'], ['VB', 'DT', 'NN', 'RB'], ['VB', 'NN', 'RB']

# 8-L4a	[action1] it [position1] to this; move it to the left of this object, move that object to the front of this object, etc;
# Patterns: ['VB', 'PRP', 'IN', 'DT'], ['VB', 'DT', 'IN', 'DT'], ['VB', 'PRP', 'IN', 'PRP'], ['VB', 'PRP', 'IN', 'PRP'], 
# ['VB', 'DT', 'IN', 'DT'], ['VB', 'PRP', 'IN', 'DT'], ['VB', 'PRP', 'IN', 'PRP'], ['VB', 'NN', 'DT'], ['VB', 'NN', 'PRP']

# 9-L4b	[action1] it [position1] this [feature1] object: move this behind that red object, move this object to the left of that green thing, etc;
# Patterns: ['VB', 'PRP', 'IN', 'DT', 'JJ', 'NN'], ['VB', 'DT', 'IN', 'DT', 'JJ', 'NN'], ['VB', 'DT', 'NN','IN', 'DT', 'JJ', 'NN'], 
#  ['VB', 'PRP', 'IN', 'DT', 'JJ', 'NN'], ['VB', 'PRP', 'IN', 'PRP'], ['VB', 'NN', 'DT'], ['VB', 'NN', 'PRP']

# 10-L5a	First [action1] this then [action2] it here; ex: first, pick up this object and then move it over there -  pick up that object, then move it here, etc;
# Patterns: L1 + L3

# 11-L5b	First [action1] an object similar in [feature_class1] to this then move it [position1] to this object; ex: first, pick up an object similar to this one in size, then move it in front of that object; pick up something in a similar color to this, and place it to the right of this object; etc;
# Patterns: L2d + L3

# 12-L5c	First [action1] an object similar in [feature_class1] to this then move it [position] to this [feature] object; ex: first, pick up an object in similar shape to this one and place it to the left of this green object; pick up an object with size similar to this one and then put it behind that small object; etc; 
# Patterns: L2d + L2d

# 13-L5f	First [action1] it here then [action2] this [feature1] object; ex: first move that object here and then pick up this big object; move this object over there and then pick this orange thing; etc;
# Patterns: L3 + L1b

# 14-L6	Move it [position1] to the object similar to this one: Move it to the right of an object of similar size to this one; move this object to behind an object of the same color as this one; etc. 
# Patterns: ['VB', 'PRP', 'NN', 'NN', 'JJ', 'DT', 'NN'], 
# ['VB', 'DT', 'NN', 'IN', 'NN', 'WDT', 'VBZ', 'IN', 'DT', 'NN'], 
# ['VB', 'NN', 'IN', 'NN', 'WDT', 'VBZ', 'IN', 'DT', 'NN']
# ['VB', 'PRP', 'NN', 'NN', 'JJ', 'NN', 'IN', 'DT', 'NN']

# 15-L7	First [action1] here. Then [action2]; ex: Move it here and then pick it up; first, move that object here and then pick it up; etc.
# Patterns: L1+L1 ? Too izi 

# 16-L7b	First [action1] it [position] to [object]. Then pick it up.; ex: First, move it to the left of this apple. Then Pick it up; Move this object to behind that bowl, then pick it up; etc. 
# Patterns: L2c + L1

#17 - unkown, use fallback