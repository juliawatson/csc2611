import argparse
from collections import defaultdict
import gensim
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

import common


def run_analogy_task(keyed_vectors, analogy_dataset, vocab):
    """
    
    """
    words = list(vocab)
    correct = 0
    for a, b, c, d_true in analogy_dataset:
        assert all([item in vocab for item in {a, b, c, d_true}])
        d_pred_vector = keyed_vectors[b] - keyed_vectors[a] + keyed_vectors[c]
        
        # sim = cosine_similarity([d_pred_vector], embedding_matrix)[0]
        distances = keyed_vectors.distances(
            d_pred_vector, other_words=words)
        d_rankings = np.argsort(distances)[:4]
        # Pick the closest word that is not one of the prompt words.
        d_pred = [words[item] for item in d_rankings
                  if words[item] not in {a, b, c}][0]
        
        if d_pred == d_true:
            correct += 1
        #print(a, b, c, d_true, "\t", d_pred)
    accuracy = correct / len(analogy_dataset)
    print(f"accuracy = {accuracy:.4f}")

    
def run_similarity_task(embedding_dict, rg65_word_pairs):
    P = list(rg65_word_pairs.keys())
    S = [rg65_word_pairs[curr_pair] for curr_pair in P]
    
    # Compute cosine similarity for word pairs, based on word2vec embeddings.
    SW2V = []
    for w1, w2 in P:
        w1_vec = embedding_dict[w1]
        w2_vec = embedding_dict[w2]
        SW2V.append(cosine_similarity([w1_vec], [w2_vec])[0][0])

    # Compare w2v similarity to human annotations.
    r, p = pearsonr(SW2V, S)
    print(f"r={r:.4f}, p={p}")

    
def part1(args):
    # Load w2v vectors and LSA vectors.
    print("Loading w2v vectors")
    model_W2V = KeyedVectors.load_word2vec_format(
        os.path.join(
            args.data_path,
            "GoogleNews-vectors-negative300.bin"),
        binary=True)
    print("Loading LSA vectors")
    M2300 = common.load_embedding_dict(
        os.path.join(args.data_path, "M2300.pickle"))
    
    M2300_vocab = set(M2300.index2word)
    w2v_vocab = set(model_W2V.index2word)
    vocab = M2300_vocab.intersection(w2v_vocab)

    rg65_words, rg65_word_pairs = common.load_rg65(
        path=os.path.join(args.data_path, "RG65.csv"),
        vocab=vocab)
    print("Running similarty task for w2v")
    run_similarity_task(model_W2V, rg65_word_pairs)
    print("Running similarty task for M2300")
    run_similarity_task(M2300, rg65_word_pairs)
    
    # Run analogy task on w2v vectors and M2300 LSA vectors.
    analogy_dataset = common.load_analogy_task(
        path=os.path.join(args.data_path, "word-test.v1.txt"),
        vocab=vocab)
    print("Running analogy task for w2v")
    run_analogy_task(model_W2V, analogy_dataset, vocab=vocab)
    print("Running analogy task for M2300")
    run_analogy_task(M2300, analogy_dataset, vocab=vocab)
    
    
def degree_of_change_cossim(
        diachronic_embeddings,
        start_year=1900, end_year=1990):
    start_year_index = diachronic_embeddings['d'].index(start_year)
    end_year_index = diachronic_embeddings['d'].index(end_year)
    result = []
    for i in range(len(diachronic_embeddings['w'])):
        first_vec = diachronic_embeddings['E'][i][start_year_index]
        last_vec = diachronic_embeddings['E'][i][end_year_index]
        result.append(1 - cosine_similarity([first_vec], [last_vec])[0][0])
    return np.array(result)


def degree_of_change_nearest_neighbors(
        diachronic_embeddings, n_neighbors=100,
        start_year=1900, end_year=1990):
    """
    This is similar to the approach from Xu & Kemp (2015). I:
    1. Compute nearest neighbors within each decade using
       similarity (they use KL divergence).
    2. Compute semantic similarity across decades by taking
       the proportion of 100 nearest neighbors in the later
       decade that were present in the earlier decade.
    """
    start_year_index = diachronic_embeddings['d'].index(start_year)
    end_year_index = diachronic_embeddings['d'].index(end_year)
    first_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, start_year_index],
        diachronic_embeddings['E'][:, start_year_index])
    last_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, end_year_index],
        diachronic_embeddings['E'][:, end_year_index])
    
    result = []
    for i in range(len(diachronic_embeddings['w'])):
        first_decade_neighbors = set(np.argsort(-first_decade_similarities[i])[:n_neighbors])
        last_decade_neighbors = set(np.argsort(-last_decade_similarities[i])[:n_neighbors])
        overlap = first_decade_neighbors.intersection(last_decade_neighbors)
        curr_sim = len(overlap) / n_neighbors
        result.append(1 - curr_sim)            
    return np.array(result)


def degree_of_change_neighbor_correlation(
        diachronic_embeddings, start_year=1900, end_year=1990):
    """
    """
    start_year_index = diachronic_embeddings['d'].index(start_year)
    end_year_index = diachronic_embeddings['d'].index(end_year)
    first_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, start_year_index],
        diachronic_embeddings['E'][:, start_year_index])
    last_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, end_year_index],
        diachronic_embeddings['E'][:, end_year_index])
    
    result = []    
    for i in range(len(diachronic_embeddings['w'])):
        curr_sim = 1 - pearsonr(
            first_decade_similarities[i], last_decade_similarities[i])[0]
        result.append(curr_sim)
            
    return np.array(result)


def get_start_year(word, diachronic_embeddings):
    
    word_index = diachronic_embeddings['w'].index(word)
    start_year = 1900
    for i, curr_start_year in enumerate(diachronic_embeddings['d']):
        curr_embedding = np.sum(diachronic_embeddings['E'][word_index][i])
        if np.all(curr_embedding == np.zeros_like(curr_embedding)):
            start_year = curr_start_year + 10
    return start_year


def detect_change_point(word, sim_func, diachronic_embeddings):
    start_year = get_start_year(word, diachronic_embeddings)
    start_year_index = diachronic_embeddings['d'].index(start_year)
    print(start_year)
    decades = diachronic_embeddings['d'][start_year_index:]
    
    distances = []
    for end_decade in decades[1:]:
        curr_distances = sim_func(
            diachronic_embeddings, start_year=start_year, end_year=end_decade)            
        distances.append(
            curr_distances[diachronic_embeddings['w'].index(word)])
    
    # for word, distances in distances_from_1900.items():
    print(word, distances)

    # TODO: make visualization
    best_score = 0
    best_change_point = 0
    for change_point_i in range(1, len(decades) - 1):
        before_change_avg = np.mean(distances[:change_point_i + 1])
        after_change_avg = np.mean(distances[change_point_i:])
        curr_change = np.abs(after_change_avg - before_change_avg)
        if curr_change > best_score:
            best_score = curr_change
            best_change_point = decades[change_point_i]
        print(f"change_point={change_point_i}, change={curr_change}")
    print(f"word={word}; change_point={best_change_point}; score={best_score}")
    plt.figure()
    plt.plot(diachronic_embeddings['d'][start_year_index + 1:], distances)
    plt.axvline(best_change_point, color="blue")
    plt.savefig(f"{word}_change_point.png")


def part2(args):
    print("Running part 2")
    
    diachronic_embeddings = common.load_diachronic_embeddings(
        os.path.join(args.data_path, "data.pkl"))

    # Compute degree of semantic change using 3 different methods.
    func_names = ["cos_sim", "neighbor_correlation", "nearest_neighbors"]
    functions = [degree_of_change_cossim,
                 degree_of_change_neighbor_correlation,
                 degree_of_change_nearest_neighbors]
    sim_results = [curr_func(diachronic_embeddings) 
                   for curr_func in functions]
    
    # There are 4 embeddings that are all zeros in the earliest decade.
    # Because semantic change for these vectors is undefined, I exclude
    # them from the lists of most/least changing words, and I do not 
    # consider them when computing correlations between methods.
    non_zero_embeddings = np.where(
        np.sum(diachronic_embeddings['E'][:, 0], axis=1) != 0)
    zero_embeddings = np.where(
        np.sum(diachronic_embeddings['E'][:, 0], axis=1) == 0)[0]
    zero_words = [diachronic_embeddings['w'][int(i)] for i in zero_embeddings]
    
    # Report the top 20 most and least changing words in table(s) from each measure.
    for method_name, curr_result in zip(func_names, sim_results):
        most_changing = [
            diachronic_embeddings['w'][i] for i in np.argsort(-curr_result)
            if i in set(non_zero_embeddings[0])][:20]
        least_changing = [
            diachronic_embeddings['w'][i] for i in np.argsort(curr_result)
            if i in set(non_zero_embeddings[0])][:20]
        print(method_name)
        print(f"\tmost changing: {most_changing}")
        print(f"\tleast changing: {least_changing}")
    
    # Measure the intercorrelations (of semantic change in all words, given the embeddings
    # from Step 1) among the three methods you have proposed and summarize the Pearson
    # correlations in a 3-by-3 table.
    print(f"Correlations for: {func_names}")
    print([
        [pearsonr(results_a[non_zero_embeddings], results_b[non_zero_embeddings])
         for results_a in sim_results]
        for results_b in sim_results])
    
    # Propose and justify a procedure for evaluating the accuracy of the methods you
    # have proposed in Step 2, and then evaluate the three methods following this proposed
    # procedure and report Pearson correlations or relevant test statistics.
    eval_words, change_ratings = common.load_semantic_change_OED()
    for method_name, curr_result in zip(func_names, sim_results):
        print(len(curr_result))
        predictions = [
            curr_result[diachronic_embeddings['w'].index(curr_word)]
            for curr_word in eval_words]
        r, pval = pearsonr(change_ratings, predictions)
        print(f"{method_name}: r={r:.4f}, p={pval}")
    
    # Repeat the analysis using randomly selected words.
    eval_words, change_ratings = common.load_semantic_change_OED(
        path="data/OED_sense_change_random.csv")
    for method_name, curr_result in zip(func_names, sim_results):
        print(len(curr_result))
        predictions = [
            curr_result[diachronic_embeddings['w'].index(curr_word)]
            for curr_word in eval_words]
        r, pval = pearsonr(change_ratings, predictions)
        print(f"{method_name}: r={r:.4f}, p={pval}")

    # Extract the top 3 changing words using the best method from Steps 2 and 3.
    # Propose and implement a simple way of detecting the point(s) of semantic change in
    # each word based on its diachronic embedding time course—visualize the time course and
    # the detected change point(s).
    # TODO: update this to pick the top 3 words of the best method.
    for word in ['programs', 'objectives', 'computer']:
        detect_change_point(
            word, degree_of_change_cossim, diachronic_embeddings)
    

def main(args):
    # part1(args)
    part2(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process each .")
    parser.add_argument(
        "--data_path",
        default="/ais/hal9000/jwatson/csc2611/data")
    
    args = parser.parse_args()
    main(args)
