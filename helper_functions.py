positional_nouns = ['right', 'left', 'front', 'behind']
banned_tokens = [',', '.', '!', '?', ':', ';', 'the', 'a', 'an', 'as', 'and', 'tiago', 'robot', 'please', 'up', 'to', 'with', 'of', 'hello', 'hi', 'first', 'then']
banned_verbs = ['color', 'is', 'are']
features = ['color', 'size', 'shape', 'format']
comparison = ['similar', 'same', 'like', 'identical']

def object_extraction(sentence):
    objects = []
    for ind, entity in enumerate(sentence):
        text = entity.to_dict()['text']
        if text not in banned_tokens + positional_nouns + features:
            if entity.to_dict()['labels'][0]['value'] in ['DT', 'PRP'] :
                if ind+1 <= len(sentence)-1:
                    if sentence[ind+1].to_dict()['labels'][0]['value'] not in ['NN', 'NNS', 'JJ']:
                        objects.append(text)
                    else:
                        pass
            
                else: 
                    objects.append(text)
            else: 
                if 'NN' in entity.to_dict()['labels'][0]['value']:
                    objects.append(text)
    return objects


def characteristics_extraction(tokens, labels, objects):
    obj_set = set(objects)
    objects = [(ind, item) for obj in obj_set for ind, item in enumerate(tokens) if item == obj]
    objects.sort(key=lambda obj: obj[0]) 
    print("Unordered: ", objects)

    characteristics = []
    for cursor_position, obj in objects:
        chars = []
        while cursor_position != 0:
            cursor_position-=1
            if labels[cursor_position]!="JJ" or tokens[cursor_position] in comparison:
                break
            else:
                chars.append(tokens[cursor_position])
            
        characteristics.append([obj, chars[::-1]])
    return characteristics


def relationship_extraction(tokens, labels, objects):
    obj_set = set(objects)
    objects = [(ind, item) for obj in obj_set for ind, item in enumerate(tokens) if item == obj]

    relationships = []
    for cursor_position, obj in objects:
        rels = []
        while cursor_position != 0:
            cursor_position-=1
            if labels[cursor_position]=="IN":
                if tokens[cursor_position].lower() == "in":
                    if tokens[cursor_position+1].lower() == "front":
                        pass
                    else: 
                        rels.append(tokens[cursor_position])
                else: 
                    rels.append(tokens[cursor_position])
            elif "NN" in labels[cursor_position] and tokens[cursor_position].lower() in positional_nouns+features:
                rels.append(tokens[cursor_position])
            
        relationships.append([obj, rels[::-1]])
    
    return relationships


def sentence_separator(tokens, labels):
    sentences = []

    verb_counter = 0
    phrase = ""

    for ind, t in enumerate(tokens): 
        
        print(labels[ind])
        if "VB" in labels[ind] and t not in banned_verbs:
            verb_counter+=1

        if verb_counter>1:
            verb_counter = 1
            sentences.append(phrase)
            phrase = t+" "
        else:
            phrase += t+" "
        print(verb_counter)

    if phrase != "":
        sentences.append(phrase)
    return sentences 


def convert_to_JSON(actions, objects, characteristics, relationships):
    json_string = "{"
    json_string += f'"action": "{actions[0]}", '
    for index, obj in enumerate(objects): 
        counter = str(index) if index>0 else ''
        json_string += f'"object{counter}":"{obj}", "characteristics{counter}": {characteristics[index][1]}, "relationship{counter}": {relationships[index][1]},'
    json_string = json_string[:-1]+"}"
    json_string = json_string.replace("[]",'null').replace("'", '"') 
    return json_string 