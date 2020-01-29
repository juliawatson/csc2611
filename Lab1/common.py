import csv
import pickle


def load_rg65(path="data/RG65.csv"):
    words_to_sim = {}
    words = set()
    with open(path) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            words_to_sim[(row["w1"], row["w2"])] = float(row["sim"])
            words.update([row["w1"], row["w2"]])
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
    with open(pickle_path, 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict
