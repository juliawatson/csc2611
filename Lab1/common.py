import csv
from gensim.models import KeyedVectors
import numpy as np
import pickle


def load_rg65(path="data/RG65.csv", vocab=None):
    words_to_sim = {}
    words = set()
    with open(path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            w1, w2 = row["w1"], row["w2"]
            if vocab is not None and not (w1 in vocab and w2 in vocab):
                continue
            words_to_sim[(w1, w2)] = float(row["sim"])
            words.update([w1, w2])
    return words, words_to_sim


def load_analogy_task(path="data/word-test.v1.txt", vocab=None):
    result = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(":") or line.startswith("//"):
                continue
            line = line.strip().split()
            if vocab is not None and not all([word in vocab for word in line]):
                continue
            result.append(line)
    return result


def write_embedding_dict(output_path, embedding_matrix, word_to_index):
    embedding_dict = {word: embedding_matrix[index]
                      for word, index in word_to_index.items()}
    with open(output_path, 'wb') as outf:
        pickle.dump(embedding_dict, outf, protocol=pickle.HIGHEST_PROTOCOL)


def load_embedding_dict(pickle_path):
    """
    returns gensim KeyedVectors
    """
    with open(pickle_path, 'rb') as f:
        embedding_dict = pickle.load(f)
        
    words = [w for w in embedding_dict]
    vectors = [embedding_dict[w] for w in words]
    embedding_dict = KeyedVectors(len(vectors[0]))
    embedding_dict.add(words, vectors)
    
    return embedding_dict


def load_diachronic_embeddings(pickle_path):
    """
    Returns a dict with 3 keys {'w', 'd', 'E'}, where:
      - 'w' maps to a list of str of length 2000. This is the list
        of words we have diachronic embedding for.
      - 'd' maps to a list of int of length 10. This indicates
        decades we have diachronic embeddings for.
      - 'E' maps to an np.array with shape (2000, 10, 300), 
        corresponding to (n_words, n_decades, n_vector_dims).
    """
    with open(pickle_path, 'rb') as f:
        result = pickle.load(f)
    result['E'] = np.array(result['E'])
    return result


def load_semantic_change_OED(path="data/OED_sense_change.csv"):
    words, new_senses = [], []
    with open(path) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip headers
        for row in csv_reader:
            # print(row)
            curr_word, rank = row[:2]
            if rank == "":
                continue
            # word_to_new_senses[curr_word] = int(new_senses)
            words.append(curr_word)
            # new_senses.append(int(curr_new_senses))
            # new_senses.append(float(curr_new_senses) / float(curr_total_senses))
            # new_senses.append(min(int(curr_new_senses), 1))
            new_senses.append(int(rank))
    return words, new_senses
            

