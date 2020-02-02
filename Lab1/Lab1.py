import argparse
import gensim
from gensim.models import KeyedVectors
import numpy as np
import os
from scipy.stats import pearsonr #, entropy
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

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

    print("Running similarty task for w2v")
    rg65_words, rg65_word_pairs = common.load_rg65(
        path=os.path.join(args.data_path, "RG65.csv"),
        vocab=vocab)
    run_similarity_task(model_W2V, rg65_word_pairs)
    
    # Run analogy task on w2v vectors and M2300 LSA vectors.
    analogy_dataset = common.load_analogy_task(
        path=os.path.join(args.data_path, "word-test.v1.txt"),
        vocab=vocab)
    print("Running analogy task for w2v")
    run_analogy_task(model_W2V, analogy_dataset, vocab=vocab)
    print("Running similarty task for M2300")
    run_analogy_task(M2300, analogy_dataset, vocab=vocab)
    
    
def degree_of_change_cossim(diachronic_embeddings):
    result = []
    for i in range(len(diachronic_embeddings['w'])):
        first_vec = diachronic_embeddings['E'][i][0]
        last_vec = diachronic_embeddings['E'][i][-1]
        result.append(cosine_similarity([first_vec], [last_vec])[0][0])
    return np.array(result)


def degree_of_change_nearest_neighbors(diachronic_embeddings, n_neighbors=100):
    """
    This is similar to the approach from Xu & Kemp (2015). I:
    1. Compute nearest neighbors within each decade using
       similarity (they use KL divergence).
    2. Compute semantic similarity across decades by taking
       the proportion of 100 nearest neighbors in the later
       decade that were present in the earlier decade.
    """
    first_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, 0],
        diachronic_embeddings['E'][:, 0])
    last_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, -1],
        diachronic_embeddings['E'][:, -1])
    
    result = []
    for i in range(len(diachronic_embeddings['w'])):
        first_decade_neighbors = set(np.argsort(-first_decade_similarities[i])[:n_neighbors])
        last_decade_neighbors = set(np.argsort(-last_decade_similarities[i])[:n_neighbors])
        overlap = first_decade_neighbors.intersection(last_decade_neighbors)
        curr_sim = len(overlap) / n_neighbors
        result.append(curr_sim)            
    return np.array(result)


def degree_of_change_neighbor_correlation(diachronic_embeddings):
    """
    """
    first_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, 0],
        diachronic_embeddings['E'][:, 0])
    last_decade_similarities = cosine_similarity(
        diachronic_embeddings['E'][:, -1],
        diachronic_embeddings['E'][:, -1])
    
    result = []    
    for i in range(len(diachronic_embeddings['w'])):
        curr_sim = pearsonr(
            first_decade_similarities[i], last_decade_similarities[i])[0]
        
        # There are 4 words for which the embeddings in the first decade
        # are constant (all zeros). In these cases, the similarity is 
        # undefined.
#         if np.isnan(curr_sim):
#             print(curr_sim)
#             print(diachronic_embeddings['w'][i])
#             print(first_decade_similarities[i])
#             print(last_decade_similarities[i])
            
#             result.append(0)
#         else:
        result.append(curr_sim)
            
    return np.array(result)


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
    
    # Report the top 20 most and least changing words in table(s) from each measure.
    for method_name, curr_result in zip(func_names, sim_results):
        most_changing = [
            diachronic_embeddings['w'][i] for i in np.argsort(curr_result)
            if i in set(non_zero_embeddings[0])][:20]
        least_changing = [
            diachronic_embeddings['w'][i] for i in np.argsort(-curr_result)
            if i in set(non_zero_embeddings[0])][:20]
        print(method_name)
        print(f"\tmost changing: {most_changing}")
        print(f"\tleast changing: {least_changing}")
    
    # Measure the intercorrelations (of semantic change in all words, given the embeddings
    # from Step 1) among the three methods you have proposed and summarize the Pearson
    # correlations in a 3-by-3 table.
    print(f"Correlations for: {func_names}")
    print(np.array([
        [pearsonr(results_a[non_zero_embeddings], results_b[non_zero_embeddings])[0]
         for results_a in sim_results]
        for results_b in sim_results]))
    
    # TODO:  Propose and justify a procedure for evaluating the accuracy of the methods you
    # have proposed in Step 2, and then evaluate the three methods following this proposed
    # procedure and report Pearson correlations or relevant test statistics.
    
    # TODO: Extract the top 3 changing words using the best method from Steps 2 and 3.
    # Propose and implement a simple way of detecting the point(s) of semantic change in
    # each word based on its diachronic embedding time course—visualize the time course and
    # the detected change point(s). 
    

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
