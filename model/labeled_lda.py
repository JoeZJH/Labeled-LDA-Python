# -*- coding: utf-8 -*-

# @Author: Jiahong Zhou
# @Date: 2018-10-20
# @Email: JoeZJiahong@gmail.com
# implement of L-LDA Model(Labeled Latent Dirichlet Allocation Model)
# References:
#   i.      Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora, Daniel Ramage...
#   ii.     Parameter estimation for text analysis, Gregor Heinrich.
#   iii.    Latent Dirichlet Allocation, David M. Blei, Andrew Y. Ng...
import numpy
import numpy as np
import os
import json
from concurrent import futures
try:
    import copy_reg
except Exception:
    import copyreg as copy_reg

import types


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class LldaModel:
    """
    L-LDA(Labeled Latent Dirichlet Allocation Model)

    @field K: the number of topics
    @field alpha_vector: the prior distribution of theta_m
                         str("50_div_K"): means [K/50, K/50, ...],
                                this value come from Parameter estimation for text analysis, Gregor Heinrich.
                         int or float: means [alpha_vector, alpha_vector, ...]
                         None: means [0.001, 0.001, ...]
    @field eta_vector: the prior distribution of beta_k
                       int or float: means [eta_vector, eta_vector, ...]
                       None: means [0.001, 0.001, ...]
    @field terms: a list of the all terms
    @field vocabulary: a dict of <term, term_id>, vocabulary[terms[id]] == id
    @field topics: a list of the all topics
    @field topic_vocabulary: a dict of <topic, topic_id>, topic_vocabulary[topics[id]] == id
    @field W: the corpus, a list of terms list,
              W[m] is the document vector, W[m][n] is the id of the term
    @field Z: the topic corpus, just same as W,
              except Z[m][n] is the id of the topic of the term
    @field M: the number of documents
    @field T: the number of terms
    @field WN: the number of all words in W
    @field LN: the number of all original labels
    @field iteration: the times of iteration
    @field all_perplexities: a list of all perplexities (one training iteration one perplexity)
    @field last_beta: the parameter `beta` of last training iteration
    @field Lambda: a matrix, shape is M * K,
                   Lambda[m][k] is 1 means topic k is a label of document m

    # derivative fields
    @field Doc2TopicCount: a matrix, shape is M * K,
                           Doc2TopicCount[m][k] is the times of topic k sampled in document m
    @field Topic2TermCount: a matrix, shape is K * T,
                            Topic2TermCount[k][t] is the times of term t generated from topic k
    @field Doc2TopicCountSum: a vector, shape is M, self.Doc2TopicCount.sum(axis=1)
                              Doc2TopicCountSum[m] is the count of all topic,
                              i.e., Doc2TopicCountSum[m] is the number of words in document m
    @field alpha_vector_Lambda: a matrix, self.alpha_vector * self.Lambda
    @field alpha_vector_Lambda_sum: a vector, self.alpha_vector_Lambda.sum(axis=1)
    @field eta_vector_sum: float value, sum(self.eta_vector)
    @field Topic2TermCountSum: a vector, self.Topic2TermCount.sum(axis=1)

    """
    def __init__(self, alpha_vector="50_div_K", eta_vector=None, labeled_documents=None):
        """

        :param alpha_vector: the prior distribution of theta_m
        :param eta_vector: the prior distribution of beta_k
        :param labeled_documents: a iterable of tuple(doc, iterable of label), contains all doc and their labels
        """
        self.alpha_vector = alpha_vector
        self.eta_vector = eta_vector
        self.terms = []
        self.vocabulary = {}
        self.topics = []
        self.topic_vocabulary = {}
        self.W = []
        self.Z = []
        self.K = 0
        self.M = 0
        self.T = 0
        self.WN = 0
        self.LN = 0
        self.iteration = 0
        self.all_perplexities = []
        self.last_beta = None
        self.Lambda = None

        # derivative fields:
        # the following fields could reduce operations in training and inference
        # it is not necessary to save them to file, we can recover them by other fields

        self.Doc2TopicCount = None
        self.Topic2TermCount = None
        # self.Doc2TopicCountSum = None
        self.alpha_vector_Lambda = None
        # self.alpha_vector_Lambda_sum = None
        self.eta_vector_sum = 0.0
        self.Topic2TermCountSum = None

        if labeled_documents is not None:
            self._load_labeled_documents(labeled_documents)

        pass

    def _initialize_derivative_fields(self):
        """
        initialize derivative fields
        :return: None
        """
        # TODO: Doc2TopicCount could be reduced to a smaller matrix,
        # TODO: because some vector in Doc2TopicCount will always been 0
        self.Doc2TopicCount = np.zeros((self.M, self.K), dtype=int)
        self.Topic2TermCount = np.zeros((self.K, self.T), dtype=int)
        for m in range(self.M):
            # print self.Z[m]
            for t, z in zip(self.W[m], self.Z[m]):
                k = z
                # print "[m=%s, k=%s]" % (m, k)
                # print "[k=%s, t=%s]" % (k, t)
                self.Doc2TopicCount[m, k] += 1
                self.Topic2TermCount[k, t] += 1

        # self.Doc2TopicCountSum = self.Doc2TopicCount.sum(axis=1)
        self.alpha_vector_Lambda = self.alpha_vector * self.Lambda
        # self.alpha_vector_Lambda_sum = self.alpha_vector_Lambda.sum(axis=1)
        self.eta_vector_sum = sum(self.eta_vector)
        self.Topic2TermCountSum = self.Topic2TermCount.sum(axis=1)

    def _load_labeled_documents(self, labeled_documents):
        """
        input labeled corpus, which contains all documents and their corresponding labels
        :param labeled_documents: a iterable of tuple(doc, iterable of label), contains all doc and their labels
        :return:
        """
        # self.documents = []
        all_labels = []
        all_words = []
        doc_corpus = []
        labels_corpus = []
        for document, labels in labeled_documents:
            document = LldaModel._document_preprocess(document)
            doc_words = document.split()
            doc_corpus.append(doc_words)
            if labels is None:
                labels = []
            labels.append("common_topic")
            labels_corpus.append(labels)
            all_words.extend(doc_words)
            all_labels.extend(labels)
        self.terms = list(set(all_words))
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}
        self.topics = list(set(all_labels))
        self.topic_vocabulary = {topic: index for index, topic in enumerate(self.topics)}
        self.K = len(self.topics)
        self.T = len(self.terms)
        self.W = [[self.vocabulary[term] for term in doc_words] for doc_words in doc_corpus]
        self.M = len(self.W)
        self.WN = len(all_words)
        # we appended topic "common_topic" to each doc at the beginning
        # so we need minus the number of "common_topic"
        # LN is the number of original labels
        self.LN = len(all_labels) - self.M

        self.Lambda = np.zeros((self.M, self.K), dtype=float)
        for m in range(self.M):
            if len(labels_corpus[m]) == 1:
                labels_corpus[m] = self.topics
            for label in labels_corpus[m]:
                k = self.topic_vocabulary[label]
                self.Lambda[m, k] = 1.0

        if self.alpha_vector is None:
            self.alpha_vector = [0.001 for _ in range(self.K)]
        elif type(self.alpha_vector) is str and self.alpha_vector == "50_div_K":
            self.alpha_vector = [50.0/self.K for _ in range(self.K)]
        elif type(self.alpha_vector) is float or type(self.alpha_vector) is int:
            self.alpha_vector = [self.alpha_vector for _ in range(self.K)]
        else:
            message = "error alpha_vector: %s" % self.alpha_vector
            raise Exception(message)

        if self.eta_vector is None:
            self.eta_vector = [0.001 for _ in range(self.T)]
        elif type(self.eta_vector) is float or type(self.eta_vector) is int:
            self.eta_vector = [self.eta_vector for _ in range(self.T)]
        else:
            message = "error eta_vector: %s" % self.eta_vector
            raise Exception(message)

        self.Z = []
        for m in range(self.M):
            # print "self.Lambda[m]: ", self.Lambda[m]
            numerator_vector = self.Lambda[m] * self.alpha_vector
            p_vector = 1.0 * numerator_vector / sum(numerator_vector)
            # print p_vector
            # print "p_vector: ", p_vector
            # z_vector is a vector of a document,
            # just like [2, 3, 6, 0], which means this doc have 4 word and them generated
            # from the 2nd, 3rd, 6th, 0th topic, respectively
            z_vector = [LldaModel._multinomial_sample(p_vector) for _ in range(len(self.W[m]))]
            self.Z.append(z_vector)

        self._initialize_derivative_fields()
        pass

    @staticmethod
    def _multinomial_sample(p_vector, random_state=None):
        """
        sample a number from multinomial distribution
        :param p_vector: the probabilities
        :return: a int value
        """
        if random_state is not None:
            return random_state.multinomial(1, p_vector).argmax()
        return np.random.multinomial(1, p_vector).argmax()

    def _gibbs_sample_training(self):
        """
        sample a topic(k) for each word(t) of all documents, Generate a new matrix Z
        :return: None
        """
        # TODO: the operations of addition and multiplication could be reduced, because some
        self.last_beta = self.beta
        count = 0
        for m in range(self.M):

            # doc_m_eta_vector = self.eta_vector
            # doc_m_alpha_vector = self.alpha_vector * self.Lambda[m]
            doc_m_alpha_vector = self.alpha_vector_Lambda[m]
            # assert (doc_m_alpha_vector == self.alpha_vector_Lambda[m]).all()

            # sum_doc_m_alpha_vector = sum(doc_m_alpha_vector)
            # sum_doc_m_alpha_vector = self.alpha_vector_Lambda_sum[m]
            # assert sum_doc_m_alpha_vector == self.alpha_vector_Lambda_sum[m]

            for t, z, n in zip(self.W[m], self.Z[m], range(len(self.W[m]))):
                k = z
                self.Doc2TopicCount[m, k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = self.Doc2TopicCount[m] + doc_m_alpha_vector
                # denominator_theta = sum(self.Doc2TopicCount[m]) + sum_doc_m_alpha_vector
                # denominator_theta = self.Doc2TopicCountSum[m]-1 + sum_doc_m_alpha_vector
                # assert sum(self.Doc2TopicCount[m]) == self.Doc2TopicCountSum[m]-1

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + sum(self.eta_vector)
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + self.eta_vector_sum
                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum
                # assert (self.Topic2TermCount.sum(axis=1) == self.Topic2TermCountSum).all()
                # assert sum(self.eta_vector) == self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                # theta_vector = 1.0 * numerator_theta_vector / denominator_theta
                # denominator_theta is independent with t and k, so denominator could be any value except 0
                # will set denominator_theta as 1.0
                theta_vector = numerator_theta_vector

                p_vector = beta_vector * theta_vector
                # print p_vector
                """
                for some special document m (only have one word) p_vector may be zero here, sum(p_vector) will be zero too
                1.0 * p_vector / sum(p_vector) will be [...nan...]
                so we should avoid inputting the special document 
                """
                p_vector = 1.0 * p_vector / sum(p_vector)
                # print p_vector
                sample_z = LldaModel._multinomial_sample(p_vector)
                self.Z[m][n] = sample_z

                k = sample_z
                self.Doc2TopicCount[m, k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1
                count += 1
        assert count == self.WN
        print("gibbs sample count: ", self.WN)
        self.iteration += 1
        self.all_perplexities.append(self.perplexity())
        pass

    def _gibbs_sample_inference(self, term_vector, iteration=300, times=10):
        """
        inference with gibbs sampling
        :param term_vector: the term vector of document
        :param iteration: the times of iteration until Markov chain converges
        :param times: the number of samples of the target distribution
                (one whole iteration(sample for all words) generates a sample, the )
                #times = #samples,
                after Markov chain converges, the next #times samples as the samples of the target distribution,
                we drop the samples before the Markov chain converges,
                the result is the average value of #times samples
        :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """
        doc_topic_count = np.zeros(self.K, dtype=int)
        accumulated_doc_topic_count = np.zeros(self.K, dtype=int)
        p_vector = np.ones(self.K, dtype=int)
        p_vector = p_vector * 1.0 / sum(p_vector)
        z_vector = [LldaModel._multinomial_sample(p_vector) for _ in term_vector]
        for n, t in enumerate(term_vector):
            k = z_vector[n]
            doc_topic_count[k] += 1
            self.Topic2TermCount[k, t] += 1
            self.Topic2TermCountSum[k] += 1

        # sum_doc_topic_count = sum(doc_topic_count)
        doc_m_alpha_vector = self.alpha_vector
        # sum_doc_m_alpha_vector = sum(doc_m_alpha_vector)
        for i in range(iteration+times):
            for n, t in enumerate(term_vector):
                k = z_vector[n]
                doc_topic_count[k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
                # denominator_theta = sum_doc_topic_count - 1 + sum_doc_m_alpha_vector

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + sum(self.eta_vector)
                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                # theta_vector = 1.0 numerator_theta_vector / denominator_theta
                # denominator_theta is independent with t and k, so denominator could be any value except 0
                # will set denominator_theta as 1.0
                theta_vector = numerator_theta_vector

                p_vector = beta_vector * theta_vector
                # print p_vector
                p_vector = 1.0 * p_vector / sum(p_vector)
                # print p_vector
                sample_z = LldaModel._multinomial_sample(p_vector)
                z_vector[n] = sample_z

                k = sample_z
                doc_topic_count[k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1
            if i >= iteration:
                accumulated_doc_topic_count += doc_topic_count
        # reset self.Topic2TermCount
        for n, t in enumerate(term_vector):
            k = z_vector[n]
            self.Topic2TermCount[k, t] -= 1
            self.Topic2TermCountSum[k] -= 1

        numerator_theta_vector = accumulated_doc_topic_count/times + doc_m_alpha_vector
        # denominator_theta = sum(doc_topic_count) + sum(doc_m_alpha_vector)
        denominator_theta = sum(numerator_theta_vector)
        theta_new = 1.0 * numerator_theta_vector / denominator_theta
        return theta_new

    # def _gibbs_sample_inference_multi_processors(self, term_vector, iteration=30):
    #     """
    #     inference with gibbs sampling
    #     :param term_vector: the term vector of document
    #     :param iteration: the times of iteration
    #     :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
    #              theta_new, a theta_vector, the doc-topic distribution
    #     """
    #     # print("gibbs sample inference iteration: %s" % iteration)
    #     # TODO: complete multi-processors code here
    #     # we copy all the shared variables may be modified on runtime
    #     random_state = np.random.RandomState()
    #     topic2term_count = self.Topic2TermCount.copy()
    #     topic2term_count_sum = self.Topic2TermCountSum.copy()
    #
    #     doc_topic_count = np.zeros(self.K, dtype=int)
    #     p_vector = np.ones(self.K, dtype=int)
    #     p_vector = p_vector * 1.0 / sum(p_vector)
    #     z_vector = [LldaModel._multinomial_sample(p_vector, random_state=random_state) for _ in term_vector]
    #     for n, t in enumerate(term_vector):
    #         k = z_vector[n]
    #         doc_topic_count[k] += 1
    #         topic2term_count[k, t] += 1
    #         topic2term_count_sum[k] += 1
    #
    #     # sum_doc_topic_count = sum(doc_topic_count)
    #     doc_m_alpha_vector = self.alpha_vector
    #     # sum_doc_m_alpha_vector = sum(doc_m_alpha_vector)
    #     for i in range(iteration):
    #         for n, t in enumerate(term_vector):
    #             k = z_vector[n]
    #             doc_topic_count[k] -= 1
    #             topic2term_count[k, t] -= 1
    #             topic2term_count_sum[k] -= 1
    #
    #             numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
    #             # denominator_theta = sum_doc_topic_count - 1 + sum_doc_m_alpha_vector
    #
    #             numerator_beta_vector = topic2term_count[:, t] + self.eta_vector[t]
    #             # denominator_beta = self.Topic2TermCount.sum(axis=1) + sum(self.eta_vector)
    #             denominator_beta = topic2term_count_sum + self.eta_vector_sum
    #
    #             beta_vector = 1.0 * numerator_beta_vector / denominator_beta
    #             # theta_vector = 1.0 numerator_theta_vector / denominator_theta
    #             # denominator_theta is independent with t and k, so denominator could be any value except 0
    #             # will set denominator_theta as 1.0
    #             theta_vector = numerator_theta_vector
    #
    #             p_vector = beta_vector * theta_vector
    #             # print p_vector
    #             p_vector = 1.0 * p_vector / sum(p_vector)
    #             # print p_vector
    #             sample_z = LldaModel._multinomial_sample(p_vector, random_state)
    #             z_vector[n] = sample_z
    #
    #             k = sample_z
    #             doc_topic_count[k] += 1
    #             topic2term_count[k, t] += 1
    #             topic2term_count_sum[k] += 1
    #     # reset self.Topic2TermCount
    #     # for n, t in enumerate(term_vector):
    #     #     k = z_vector[n]
    #     #     self.Topic2TermCount[k, t] -= 1
    #     #     self.Topic2TermCountSum[k] -= 1
    #
    #     numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
    #     # denominator_theta = sum(doc_topic_count) + sum(doc_m_alpha_vector)
    #     denominator_theta = sum(numerator_theta_vector)
    #     theta_new = 1.0 * numerator_theta_vector / denominator_theta
    #     return theta_new

    def training(self, iteration=10, log=False):
        """
        training this model with gibbs sampling
        :param log: print perplexity after every gibbs sampling if True
        :param iteration: the times of iteration
        :return: None
        """
        for i in range(iteration):
            if log:
                print("after iteration: %s, perplexity: %s" % (self.iteration, self.perplexity()))
            self._gibbs_sample_training()
        pass

    def inference(self, document, iteration=30, times=10):
        # TODO: inference of a document
        """
        inference for one document
        :param document: some sentence like "this is a method for inference"
        :param times: the number of samples of the target distribution
                (one whole iteration(sample for all words) generates a sample, the )
                #times = #samples,
                after Markov chain converges, the next #times samples as the samples of the target distribution,
                we drop the samples before the Markov chain converges,
                the result is the average value of #times samples
        :param iteration: the times of iteration until Markov chain converges
        :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """
        document = LldaModel._document_preprocess(document)
        doc_words = document.split()
        term_vector = [self.vocabulary[word] for word in doc_words if word in self.vocabulary]
        theta_new = self._gibbs_sample_inference(term_vector, iteration=iteration, times=times)
        doc_topic_new = [(self.topics[k], probability) for k, probability in enumerate(theta_new)]
        sorted_doc_topic_new = sorted(doc_topic_new,
                                      key=lambda topic_probability: topic_probability[1],
                                      reverse=True)
        return sorted_doc_topic_new
        pass

    # def inference_multi_processors(self, document, iteration=30, times=8, max_workers=8):
    #     # TODO: inference of a document with multi processors
    #     """
    #     inference for one document
    #     :param times: the times of gibbs sampling, the result is the average value of all times(gibbs sampling)
    #     :param iteration: the times of iteration
    #     :param document: some sentence like "this is a method for inference"
    #     :param max_workers: the max number of processors(workers)
    #     :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
    #              theta_new, a theta_vector, the doc-topic distribution
    #     """
    #
    #     def _pickle_method(m):
    #         if m.im_self is None:
    #             return getattr, (m.im_class, m.im_func.func_name)
    #         else:
    #             return getattr, (m.im_self, m.im_func.func_name)
    #     copy_reg.pickle(types.MethodType, _pickle_method)
    #
    #     words = document.split()
    #     term_vector = [self.vocabulary[word] for word in words if word in self.vocabulary]
    #     term_vectors = [term_vector for _ in range(times)]
    #     iterations = [iteration for _ in range(times)]
    #
    #     with futures.ProcessPoolExecutor(max_workers) as executor:
    #         # print "executor.map"
    #         res = executor.map(self._gibbs_sample_inference_multi_processors, term_vectors, iterations)
    #     theta_new_accumulation = np.zeros(self.K, float)
    #     for theta_new in res:
    #         theta_new_accumulation += theta_new
    #     theta_new = 1.0 * theta_new_accumulation / times
    #     # print "avg: \n", theta_new
    #     doc_topic_new = [(self.topics[k], probability) for k, probability in enumerate(theta_new)]
    #     sorted_doc_topic_new = sorted(doc_topic_new,
    #                                   key=lambda topic_probability: topic_probability[1],
    #                                   reverse=True)
    #     return sorted_doc_topic_new
    #     pass

    def beta_k(self, k):
        """
        topic-term distribution
        beta_k[t] is the probability of term t(word) to be generated from topic k
        :return: a vector, shape is T
        """
        numerator_vector = self.Topic2TermCount[k] + self.eta_vector
        # denominator = sum(self.Topic2TermCount[k]) + sum(self.eta_vector)
        denominator = sum(numerator_vector)
        return 1.0 * numerator_vector / denominator

    def theta_m(self, m):
        """
        doc-topic distribution
        theta_m[k] is the probability of doc m to be generated from topic k
        :return: a vector, shape is K
        """
        numerator_vector = self.Doc2TopicCount[m] + self.alpha_vector * self.Lambda[m]
        # denominator = sum(self.Doc2TopicCount[m]) + sum(self.alpha_vector * self.Lambda[m])
        denominator = sum(numerator_vector)
        return 1.0 * numerator_vector / denominator

    @property
    def beta(self):
        """
        This name "beta" comes from
            "Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora, Daniel Ramage..."
        topic-term distribution
        beta[k, t] is the probability of term t(word) to be generated from topic k
        :return: a matrix, shape is K * T
        """
        numerator_matrix = self.Topic2TermCount + self.eta_vector
        # column vector
        # denominator_vector = self.Topic2TermCount.sum(axis=1).reshape(self.K, 1) + sum(self.eta_vector)
        denominator_vector = numerator_matrix.sum(axis=1).reshape(self.K, 1)
        return 1.0 * numerator_matrix / denominator_vector

        pass

    @property
    def theta(self):
        """
        doc-topic distribution
        theta[m, k] is the probability of doc m to be generated from topic k
        :return: a matrix, shape is M * K
        """
        numerator_matrix = self.Doc2TopicCount + self.alpha_vector * self.Lambda
        denominator_vector = numerator_matrix.sum(axis=1).reshape(self.M, 1)
        # column vector
        return 1.0 * numerator_matrix / denominator_vector
        pass

    def log_perplexity(self, documents=None, iteration=30, times=10):
        """
        log perplexity of LDA topic model, use the training data if documents is None
        Reference:  Parameter estimation for text analysis, Gregor Heinrich.
        :param: documents: test set
        :return: a float value
        """
        beta, theta, W, WN, log_likelihood = self.beta, None, None, None, 0
        # theta is the doc-topic distribution matrix
        # W is the list of term_vector, each term_vector represents a document
        # WN is the number of all word in W
        # difference test set means difference theta, W, WN

        if not documents:
            theta = self.theta
            W = self.W
            WN = self.WN
        else:
            # generate the term_vector of document
            documents = [LldaModel._document_preprocess(document) for document in documents]
            test_corpus = [document.split() for document in documents]
            W = [[self.vocabulary[term] for term in doc_words if term in self.vocabulary] for doc_words in test_corpus]
            WN = sum([len(term_vector) for term_vector in W])
            theta = []
            for term_vector in W:
                # sample on term_vector until Markov chain converges
                theta_new = self._gibbs_sample_inference(term_vector, iteration=iteration, times=times)
                theta.append(theta_new)

        # caculate the log_perplexity of current documents
        for m, theta_m in enumerate(theta):
            for t in W[m]:
                likelihood_t = np.inner(theta_m, beta[:, t])
                log_likelihood += -np.log(likelihood_t)
        return 1.0 * log_likelihood / WN

    def perplexity(self, documents=None, iteration=30, times=10):
        """
        perplexity of LDA topic model, we use the training data if documents is None
        Reference:  Parameter estimation for text analysis, Gregor Heinrich.
        :param: documents: test set
        :return: a float value, perplexity = exp{log_perplexity}
        """
        return np.exp(self.log_perplexity(documents=documents, iteration=iteration, times=times))

    def __repr__(self):
        return "\nLabeled-LDA Model:\n" \
               "\tK = %s\n" \
               "\tM = %s\n" \
               "\tT = %s\n" \
               "\tWN = %s\n" \
               "\tLN = %s\n" \
               "\talpha = %s\n" \
               "\teta = %s\n" \
               "\tperplexity = %s\n" \
               "\t" % (self.K, self.M, self.T, self.WN, self.LN, self.alpha_vector[0], self.eta_vector[0],
                       self.perplexity())
        pass

    class SaveModel:
        def __init__(self, save_model_dict=None):
            self.alpha_vector = []
            self.eta_vector = []
            self.terms = []
            self.vocabulary = {}
            self.topics = []
            self.topic_vocabulary = {}
            self.W = []
            self.Z = []
            self.K = 0
            self.M = 0
            self.T = 0
            self.WN = 0
            self.LN = 0
            self.iteration = 0

            # the following fields cannot be dumped into json file
            # we need write them with np.save() and read them with np.load()
            # self.Doc2TopicCount = None
            # self.Topic2TermCount = None
            self.Lambda = None

            if save_model_dict is not None:
                self.__dict__ = save_model_dict
        pass

    @staticmethod
    def _document_preprocess(document):
        """
        process document before inputting it into the model(both training, update and inference)
        :param document: the target document
        :return: the word we change
        """
        document = document.lower()
        return document

    @staticmethod
    def _read_object_from_file(file_name):
        """
        read an object from json file
        :param file_name: json file name
        :return: None if file doesn't exist or can not convert to an object by json, else return the object
        """
        if os.path.exists(file_name) is False:
            print ("Error read path: [%s]" % file_name)
            return None
        with open(file_name, 'r') as f:
            try:
                obj = json.load(f)
            except Exception:
                print ("Error json: [%s]" % f.read()[0:10])
                return None
        return obj

    @staticmethod
    def _write_object_to_file(file_name, target_object):
        """
        write the object to file with json(if the file exists, this function will overwrite it)
        :param file_name: the name of new file
        :param target_object: the target object for writing
        :return: True if success else False
        """
        dirname = os.path.dirname(file_name)
        LldaModel._find_and_create_dirs(dirname)
        try:
            with open(file_name, "w") as f:
                json.dump(target_object, f, skipkeys=False, ensure_ascii=False, check_circular=True, allow_nan=True,
                          cls=NpEncoder, indent=True, separators=None, default=None, sort_keys=False)
        except Exception as e:
            message = "Write [%s...] to file [%s] error: json.dump error" % (str(target_object)[0:10], file_name)
            print ("%s: %s" % (e, message))
            return False
        else:
            # print ("Write %s" % file_name)
            return True

    @staticmethod
    def _find_and_create_dirs(dir_name):
        """
        find dir, create it if it doesn't exist
        :param dir_name: the name of dir
        :return: the name of dir
        """
        if os.path.exists(dir_name) is False:
            os.makedirs(dir_name)
        return dir_name

    def save_model_to_dir(self, dir_name, save_derivative_properties=False):
        """
        save model to directory dir_name
        :param save_derivative_properties: save derivative properties if True
            some properties are not necessary save to disk, they could be derived from some basic properties,
            we call they derivative properties.
            to save derivative properties to disk:
                it will reduce the time of loading model from disk (read properties directly but do not compute them)
                but, meanwhile, it will take up more disk space
        :param dir_name: the target directory name
        :return: None
        """
        save_model = LldaModel.SaveModel()
        save_model.alpha_vector = self.alpha_vector
        save_model.eta_vector = self.eta_vector
        save_model.terms = self.terms
        save_model.vocabulary = self.vocabulary
        save_model.topics = self.topics
        save_model.topic_vocabulary = self.topic_vocabulary
        save_model.W = self.W
        save_model.Z = self.Z
        save_model.K = self.K
        save_model.M = self.M
        save_model.T = self.T
        save_model.WN = self.WN
        save_model.LN = self.LN
        save_model.iteration = self.iteration

        save_model_path = os.path.join(dir_name, "llda_model.json")
        LldaModel._write_object_to_file(save_model_path, save_model.__dict__)

        np.save(os.path.join(dir_name, "Lambda.npy"), self.Lambda)
        # save derivative properties
        if save_derivative_properties:
            np.save(os.path.join(dir_name, "Doc2TopicCount.npy"), self.Doc2TopicCount)
            np.save(os.path.join(dir_name, "Topic2TermCount.npy"), self.Topic2TermCount)
            np.save(os.path.join(dir_name, "alpha_vector_Lambda.npy"), self.alpha_vector_Lambda)
            np.save(os.path.join(dir_name, "eta_vector_sum.npy"), self.eta_vector_sum)
            np.save(os.path.join(dir_name, "Topic2TermCountSum.npy"), self.Topic2TermCountSum)
        pass

    def load_model_from_dir(self, dir_name, load_derivative_properties=True):
        """
        load model from directory dir_name
        :param load_derivative_properties: load derivative properties from disk if True
        :param dir_name: the target directory name
        :return: None
        """
        save_model_path = os.path.join(dir_name, "llda_model.json")
        save_model_dict = LldaModel._read_object_from_file(save_model_path)
        save_model = LldaModel.SaveModel(save_model_dict=save_model_dict)
        self.alpha_vector = save_model.alpha_vector
        self.eta_vector = save_model.eta_vector
        self.terms = save_model.terms
        self.vocabulary = save_model.vocabulary
        self.topics = save_model.topics
        self.topic_vocabulary = save_model.topic_vocabulary
        self.W = save_model.W
        self.Z = save_model.Z
        self.K = save_model.K
        self.M = save_model.M
        self.T = save_model.T
        self.WN = save_model.WN
        self.LN = save_model.LN
        self.iteration = save_model.iteration

        self.Lambda = np.load(os.path.join(dir_name, "Lambda.npy"))

        # load load_derivative properties
        if load_derivative_properties:
            try:
                self.Doc2TopicCount = np.load(os.path.join(dir_name, "Doc2TopicCount.npy"))
                self.Topic2TermCount = np.load(os.path.join(dir_name, "Topic2TermCount.npy"))
                self.alpha_vector_Lambda = np.load(os.path.join(dir_name, "alpha_vector_Lambda.npy"))
                self.eta_vector_sum = np.load(os.path.join(dir_name, "eta_vector_sum.npy"))
                self.Topic2TermCountSum = np.load(os.path.join(dir_name, "Topic2TermCountSum.npy"))
            except IOError or ValueError as e:
                print("%s: load derivative properties fail, initialize them with basic properties" % e)
                self._initialize_derivative_fields()
        else:
            self._initialize_derivative_fields()
        pass

    def update(self, labeled_documents=None):
        """
        update model with labeled documents, incremental update
        :return: None
        """
        self.all_perplexities = []
        if labeled_documents is None:
            pass

        new_labels = []
        new_words = []
        new_doc_corpus = []
        new_labels_corpus = []
        for document, labels in labeled_documents:
            document = LldaModel._document_preprocess(document)
            doc_words = document.split()
            new_doc_corpus.append(doc_words)
            if labels is None:
                labels = []
            labels.append("common_topic")
            new_labels_corpus.append(labels)
            new_words.extend(doc_words)
            new_labels.extend(labels)
        # self.terms = list(set(new_words))
        new_terms = set(new_words) - set(self.terms)
        self.terms.extend(new_terms)
        self.vocabulary = {term: index for index, term in enumerate(self.terms)}

        # self.topics = list(set(new_labels))
        new_topics = set(new_labels) - set(self.topics)
        self.topics.extend(new_topics)
        self.topic_vocabulary = {topic: index for index, topic in enumerate(self.topics)}

        old_K = self.K
        old_T = self.T
        self.K = len(self.topics)
        self.T = len(self.terms)

        # self.W = [[self.vocabulary[term] for term in doc_words] for doc_words in new_doc_corpus]
        new_w_vectors = [[self.vocabulary[term] for term in doc_words] for doc_words in new_doc_corpus]
        for new_w_vector in new_w_vectors:
            self.W.append(new_w_vector)

        old_M = self.M
        old_WN = self.WN
        self.M = len(self.W)
        self.WN += len(new_words)
        # we appended topic "common_topic" to each doc at the beginning
        # so we need minus the number of "common_topic"
        # LN is the number of original labels
        old_LN = self.LN

        self.LN += len(new_labels) + len(new_labels_corpus)

        old_Lambda = self.Lambda
        self.Lambda = np.zeros((self.M, self.K), dtype=float)
        for m in range(self.M):
            if m < old_M:
                # if the old document has no topic, we also init it to all topics here
                if sum(old_Lambda[m]) == old_K:
                    # set all value of self.Lambda[m] to 1.0
                    self.Lambda[m] += 1.0
                continue
            # print m, old_M
            if len(new_labels_corpus[m-old_M]) == 1:
                new_labels_corpus[m-old_M] = self.topics
            for label in new_labels_corpus[m-old_M]:
                k = self.topic_vocabulary[label]
                self.Lambda[m, k] = 1.0

        # TODO: the following 2 fields should be modified again if alpha_vector is not constant vector
        self.alpha_vector = [self.alpha_vector[0] for _ in range(self.K)]
        self.eta_vector = [self.eta_vector[0] for _ in range(self.T)]

        # self.Z = []
        for m in range(old_M, self.M):
            # print "self.Lambda[m]: ", self.Lambda[m]
            numerator_vector = self.Lambda[m] * self.alpha_vector
            p_vector = numerator_vector / sum(numerator_vector)
            # print p_vector
            # print "p_vector: ", p_vector
            # z_vector is a vector of a document,
            # just like [2, 3, 6, 0], which means this doc have 4 word and them generated
            # from the 2nd, 3rd, 6th, 0th topic, respectively
            z_vector = [LldaModel._multinomial_sample(p_vector) for _ in range(len(self.W[m]))]
            self.Z.append(z_vector)

        self._initialize_derivative_fields()
        pass

    @staticmethod
    def _extend_matrix(origin=None, shape=None, padding_value=0):
        """
        for quickly extend the matrices when update
        extend origin matrix with shape, padding with padding_value
        :type shape: the shape of new matrix
        :param origin: np.ndarray, the original matrix
        :return: np.ndarray, a matrix with new shape
        """
        new_matrix = np.zeros(shape, dtype=origin.dtype)

        for row in range(new_matrix.shape[0]):
            for col in range(new_matrix.shape[1]):
                if row < origin.shape[0] and col < origin.shape[0]:
                    new_matrix[row, col] = origin[row, col]
                else:
                    new_matrix[row, col] = padding_value

        return new_matrix
        pass

    def is_convergent(self, method="PPL", delta=0.001):
        """
        is this model convergent?
        use the perplexities to determine whether the Markov chain converges
        :param method: the method of determining whether the Markov chain converges
                "PPL": use the perplexities of training data
                "beta": use the parameter 'beta'
        :param delta: if the changes are less than or equal to `delta`, means that the Markov chain converges
        :return: True if model is convergent
        """
        if method == "PPL":
            if len(self.all_perplexities) < 10:
                return False
            perplexities = self.all_perplexities[-10:]
            if max(perplexities) - min(perplexities) <= delta:
                return True
            return False
        elif method == "beta":
            if self.delta_beta <= delta:
                return True
            return False
        else:
            raise Exception("parameter 'method=\"%s\"' is illegal" % method)

    @property
    def delta_beta(self):
        """
        calculate the changes of the parameter `beta`
        :return: the sum of changes of the parameter `beta`
        """
        return np.sum(np.abs(self.beta - self.last_beta))

    def top_terms_of_topic(self, topic, k, with_probabilities=True):
        """
        get top-k terms of topic
        :param with_probabilities: True means return the probabilities of a term generated by topic,
                                   else return only terms
        :param topic: str, the name of topic
        :param k: int, the number of terms
        :return: the top-k terms of topic
        """
        if topic not in self.topic_vocabulary:
            raise Exception("Cannot find topic \"%s\"" % topic)
        beta = self.beta_k(self.topic_vocabulary[topic])
        terms = sorted(list(zip(self.terms, beta)), key=lambda x: x[1], reverse=True)
        if with_probabilities:
            return terms[:k]
        return [term for term, p in terms[:k]]


if __name__ == "__main__":
    pass

