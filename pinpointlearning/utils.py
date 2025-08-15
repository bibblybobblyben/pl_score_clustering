"""
Utility functions for creating the exam analysis tools.
"""

import numpy as np
from k_means_constrained import KMeansConstrained
from itertools import permutations, product

# pylint: disable = C0103


def calculate_bernoulli_prob(mu, X_s):
    """_summary_

    Args:
        mu (_type_): _description_
        X_s (_type_): _description_

    Returns:
        _type_: _description_
    """

    return np.power(mu, X_s) * np.power(1 - np.array(mu), 1 - X_s)


def load_sample_data(
    n_cols: int = 24, n_rows: int = 10**3, binary: bool = False, synthetic=True
):
    """Returns an array of sample data for use in prototyping the analysis
    pipelines

    Args:
        n_cols (int, optional): Number of columns to return. Columsn represent
        independent questions. Defaults to 24.
        n_rows (_type_, optional): Number of rows to return. Rows represent
         individual students. Defaults to 10**3.
        binary (bool, optional): Whether to return raw scores or binarised
        pass/fail data. Defaults to False.
        synthetic (bool, option): Whether to load the read data or generate
         random placeholder values. Defaults to False

    Returns:
        _type_: _description_
    """
    if synthetic:
        data = np.random.uniform(0, 7, (n_rows, n_cols))
    else:
        data = np.load("../data/processed/scores/Exam_1.npy")
        mask = ~np.isnan(data[:, 0])
        data = data[mask]
        data = np.nan_to_num(data)
        data = data[np.amax(data, axis=1) <= 7, :]
        data = data[:n_rows]

    if binary:
        data = data / np.amax(data, axis=0)
        data = data > 0.5
    return data


def cosine_sim(vec1, vec2):
    """For two input vectors, calculate the cosine similarity

    Args:
        vec1 (_type_): One vector to be evaluated
        vec2 (_type_): A second vector for evaluating similarity

    Returns:
        _type_: vector similarity between vec1 and vec2
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def norm_difference(vec1, vec2):
    """Calculate the normalised vector difference between two vectors

    Args:
        vec1 (_type_): first vector for evaluating distance
        vec2 (_type_): second vector for evaluating distance

    Returns:
        _type_: normalised vector distance between vec1 and vec2
    """
    return np.linalg.norm(vec1 - vec2)


def pass_threshold(v_test, v_level, v_min=True):
    """
    Tool to streamline the evaluation of whether a threshold has been exceeded.

    Args:
        v_test: Value to evaluate against a threshold
        v_level: Threshold value
        min: Whether to test for if a value is below or above a threshold. If True,
    function returns True if v_test is below v_level. If False, returns True if
    v_test> v_level. Otherwise returns False.

    Returns:

    """
    if v_min:
        if v_test < v_level:
            return True
        return False
    if v_level < v_test:
        return True
    return False


def mapping_scores(vecs_in, vecs_match, metric, mapping):
    """
    Tool to produce a list of scores (according to metric) for two lists of vectors (
    vecs_in, and vecs_match) according to a defined mapping

    Args:
        vecs_in: List of vectors to evaluate
        vecs_match: List of reference vectors to compare against
        metric: Function returning a score for the relationship between two vectors
        mapping: List of coordinates defining which vector in vecs_in to compare
        against which in vecs_match. The index of an entry in mapping matches the
        index of the vector used in vecs_in, and the number at that index defines the
        index of the vector in vecs_match to evaluate against

    Returns:

    """
    scores = []
    for i, vec in enumerate(vecs_in):
        scores.append(metric(vec, vecs_match[mapping[i]]))
    return scores


def assign_vector_mapping(
    vecs_in,
    vecs_match,
    metric=cosine_sim,
    minimise=False,
    force_unique=True,
    try_all=False,
    agg_func=np.mean,
):
    """Create a mapping between two arrays of vectors. Designed to be useful with the
    output of the EM algorithm , where no ordering is enforced. Returns a dictionary
    with keys for the position in first vector list, and value for the position in the
    second vector list.

    To enforce that each vector in vecs_in is only mapped to a single vector in
    vecs_match, use the "force_unique" argument. This will assign vectors to a single
    corresponding partner, assigning a pair across the two vector lists in the priority
    order defined using metric and minimise arguments.

    To explore all possible permutations of the mapping, use the "try_all" variable.
    The quality of the combined fit of the mapping is defined by the function passed
    to agg_func

    Args:
        vecs_in (list): List of vectors
        vecs_match (list): List of second vectors to map to vecs_in
        metric (_type_, optional): Metric to use for evaluating similarity between
        vecs_in and vecs_match. Defaults to cosine_sim.
        minimise (bool, optional): Whether the similarity metric is to be minimised or
        maximised. If True, attempts to minimise the metric outputs. Defaults to False.
        force_unique (bool, optional): Whether to force the matching algorithm to only
        allow each vector in vecs_in to only map to one vector in vecs_match.
        If False, multiple vectors can map to a single vector. Defaults to True.
        try_all (bool, optional): Whether to explore every combination of assignments
        and allocate mapping this way. If True, all possible permutations will be
        explored and the best fit returned, if False, runs an algorithm to map the most
        similar vectors
        together, then the second most similar, etc
        agg_func (Callable, optional): Function to aggregate the individual vector
        similarities into a total cost function. Defaults to calculating the mean

    Returns:
        _type_: Dict mapping {vector index in vecs_in : vector index in vecs_match}
    """
    map_dict = {}

    if try_all:
        if force_unique:
            combinations = permutations(range(len(vecs_match)))
        else:
            combinations = product(range(len(vecs_match)), repeat=len(vecs_match))

        mapping = next(combinations)
        refscore = agg_func(mapping_scores(vecs_in, vecs_match, metric, mapping))
        map_dict = {a: mapping[a] for a in range(len(vecs_in))}
        for mapping in combinations:
            scores = []
            for i, vec in enumerate(vecs_in):
                scores.append(metric(vec, vecs_match[mapping[i]]))
            score = agg_func(scores)
            if pass_threshold(score, refscore, v_min=minimise):
                refscore = score
                map_dict = {a: mapping[a] for a in range(len(vecs_in))}

    else:
        sim_matrix = np.empty((len(vecs_in), len(vecs_match)))

        for i, v1 in enumerate(vecs_in):
            for j, v2 in enumerate(vecs_match):
                sim_matrix[i, j] = metric(v1, v2)
        if minimise:
            if force_unique:
                for n in range(len(vecs_in)):
                    coords = np.where(sim_matrix == np.amin(sim_matrix))
                    map_dict[coords[0][0]] = coords[1][0]
                    sim_matrix[:, coords[1][0]] = np.inf
                    sim_matrix[coords[0][0], :] = np.inf
            else:
                map_dict[i] = np.argmin(sim_matrix)
        else:
            if force_unique:
                for n in range(len(vecs_in)):
                    coords = np.where(sim_matrix == np.amax(sim_matrix))
                    map_dict[coords[0][0]] = coords[1][0]
                    sim_matrix[:, coords[1][0]] = 0.0
                    sim_matrix[coords[0][0], :] = 0.0
            else:
                for i in range(len(vecs_in)):
                    map_dict[i] = np.argmax(sim_matrix[i, :])

    return map_dict


def match_vectors(vecs_1, vecs_2):
    """Produce one-to-one closest matching from one list of vectors to another, not
    necessarily of the same length. Uses constrained version of KNN algorithm
    where a group size of one can be specified for each cluster. "Cluster centres" are
    then fit as the longer list of vectors and the other vectors are matched to this.

    Args:
        vecs_1 (ndarray): List of vectors.
        vecs_2 (ndarray): List of vectors.

    Returns:
        dict: Mapping from one list of vectors to other.
    """

    # Number of vectors in each list of vectors
    n_vecs_1 = vecs_1.shape[0]
    n_vecs_2 = vecs_2.shape[0]

    # Use longest list of vectors as "cluster centers"
    # Constrained k-means with group size of one
    if n_vecs_1 > n_vecs_2:
        clf = KMeansConstrained(n_clusters=n_vecs_1, size_min=0, size_max=1)
        vecs_1_labels = clf.fit_predict(vecs_1)
        vecs_2_labels = clf.predict(vecs_2)
    else:
        clf = KMeansConstrained(n_clusters=n_vecs_2, size_min=0, size_max=1)
        vecs_2_labels = clf.fit_predict(vecs_2)
        vecs_1_labels = clf.predict(vecs_1)

    # K-means doesn't necessarily maintain original ordering
    # Mapping from original ordering to labels to other list of vectors
    vecs_1_to_labels = {
        vecs_1: lab for vecs_1, lab in zip(range(0, n_vecs_1), vecs_1_labels)
    }
    labels_to_vecs_2 = {
        lab: vecs_2 for lab, vecs_2 in zip(vecs_2_labels, range(0, n_vecs_2))
    }

    # Use mappings to produce mapping from one list of vectors to other
    vecs_1_to_vecs_2 = {}
    for key, value in vecs_1_to_labels.items():
        try:
            vecs_1_to_vecs_2[key] = labels_to_vecs_2[value]
        except KeyError:
            continue

    return vecs_1_to_vecs_2


def calculate_scores(vecs_1, vecs_2, vecs_map, metric=norm_difference):
    """Calculate score for each mapping

    Args:
        vecs_1 (ndarray): List of vectors.
        vecs_2 (ndarray): List of vectors.
        vecs_map (dict): Mapping from vecs_1 to vecs_2.
        metric (function): Metric choice.
    Returns:
        list: Score for each mapping
    """
    scores = []
    for vecs_1_i, vecs_2_i in vecs_map.items():
        scores.append(metric(vecs_1[vecs_1_i], vecs_2[vecs_2_i]))
    return np.array(scores)
