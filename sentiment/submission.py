#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from mpmath.math2 import sqrt2

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    x = (x.strip().lower().split())
    feature_vector = {}
    for key in x:
        feature_vector[key] = feature_vector.get(key, 0) + 1
    return feature_vector
    # raise Exception("Not implemented yet")
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes:
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and
      validationExamples to see how you're doing as you learn after each epoch.
    - The predictor should output +1 if the score is precisely 0.
    '''
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    x_trainExamples, y_trainExamples = zip(*trainExamples)
    x_validationExamples, y_validationExamples = zip(*validationExamples)

    all_keys = set()

    trainfeatureVectors = []
    validationfeatureVectors = []

    print(len(x_trainExamples))
    for example in x_trainExamples:
        trainFeatureVector = featureExtractor(example)
        trainfeatureVectors.append(trainFeatureVector)
        all_keys = all_keys.union(set(list(trainFeatureVector.keys())))

    for example in x_validationExamples:
        validationFeatureVector = featureExtractor(example)
        validationfeatureVectors.append(validationFeatureVector)
        all_keys = all_keys.union(set(list(validationFeatureVector.keys())))

    weights = {x: 0 for x in all_keys}

    for epoch in range(numEpochs):
        print(f"====== EPOCH {epoch} ======")
        total_loss = 0.0

        def dot_product(dict1, dict2):
            res = 0.0
            for key, value in dict1.items():
                res += value * dict2.get(key, 0)
            return res

        def update(dict1, dict2, lr, y):
            for key, value in dict2.items():
                dict1[key] += value * lr * y

        for (i, trainFeatureVector) in enumerate(trainfeatureVectors):
            y = y_trainExamples[i]
            loss = max(0, 1-dot_product(trainFeatureVector, weights)*y)
            if loss > 0:
                # Update w = w + lr * phi(x) * y
                update(weights, trainFeatureVector, eta, y)

            total_loss += loss
        print(f"Training loss: {total_loss/len(trainfeatureVectors)}")

        # Evaluation
        validationLoss = 0.0
        for (i, validationFeatureVector) in enumerate(validationfeatureVectors):
            y = y_validationExamples[i]
            loss = max(0, 1-dot_product(validationFeatureVector, weights)*y)
            validationLoss += loss

        print(f"Validation loss: {validationLoss/len(validationfeatureVectors)}")
    # END_YOUR_CODE
    return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified
      correctly by |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # note that there is intentionally flexibility in how you define phi.
    # y should be 1 or -1 as classified by the weight vector.
    # IMPORTANT: In the case that the score is 0, y should be set to 1.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        num_words = random.randint(1, 10)
        phi = {}
        for _ in range(num_words):
            word = random.choice(list(weights.keys()))
            phi[word] = phi.get(word, 0) + 1

        res = dotProduct(phi, weights)
        y = 1 if res >= 0 else 0
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3d: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that 1 <= n <= len(x).
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        x = x.strip().lower().replace(" ", "")
        features = {}
        for i in range(len(x)-2):
            key = x[i:i+n]
            features[key] = features.get(key, 0) + 1
        return features
        # END_YOUR_CODE
    return extract


############################################################
# Problem 3e:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 5: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    Perform K-means clustering on |examples|, where each example is a sparse feature vector.

    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    assignments = [0] * len(examples)
    clusters = [{key: random.random() for key in examples[0].keys()} for _ in range(K)]

    def SE(dict1, dict2):
        res = 0
        for key in dict1.keys():
            res += (dict1[key] - dict2.get(key, 0)) ** 2
        return res ** 0.5

    for epoch in range(maxEpochs):
        for i, example in enumerate(examples):
            # Find cluster
            minDiff = float('inf')
            for j, cluster in enumerate(clusters):
                if SE(example, cluster) < minDiff:
                    minDiff = SE(example, cluster)
                    assignments[i] = j
        total_cost = 0.0
        for i, cluster in enumerate(clusters):
            num_assignments = assignments.count(i)
            if num_assignments == 0: continue

            total_vector = {key: 0 for key in examples[0].keys()}
            for j, value in enumerate(assignments):
                if value == i:
                    def sum(dict1, dict2):
                        res = {}
                        for key in dict1.keys():
                            res[key] = dict1.get(key, 0) + dict2.get(key, 0)
                        return res

                    total_vector = sum(total_vector, examples[j])

            for key in total_vector:
                cluster[key] = total_vector[key] / num_assignments

            for j, value in enumerate(assignments):
                if value == i:
                    total_cost += SE(examples[j], clusters[i])

    return (
        clusters,
        assignments,
        total_cost
    )
    # END_YOUR_CODE
