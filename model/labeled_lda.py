# -*- coding: utf-8 -*-

# @Author: Jiahong Zhou
# @Date: 2018-10-20
# @Email: JoeZJiahong@gmail.com
# implement of L-LDA Model(Labeled Latent Dirichlet Allocation Model)
# References:
#   i.      Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora, Daniel Ramage...
#   ii.     Parameter estimation for text analysis, Gregor Heinrich.
#   iii.    Latent Dirichlet Allocation, David M. Blei, Andrew Y. Ng...

import numpy as np
import os
import json


class LldaModel:
    """
    L-LDA(Labeled Latent Dirichlet Allocation Model)

    @field K: the number of topics
    @field alpha_vector: the prior distribution of theta_m
    @field eta_vector: the prior distribution of beta_k
    @field terms: a list of the all terms
    @field vocabulary: a dict of <term, term_id>, vocabulary[terms[id]] == id
    @field topics: a list of the all topics
    @field topic_vocabulary: a dict of <topic, topic_id>, topic_vocabulary[topics[id]] == id
    @field W: the corpus, a list of terms list,
              W[m] is the document vector, W[m][n] is the id of the term
    @field Lambda: a matrix, shape is M * K,
                   Lambda[m][k] is 1 means topic k is a label of document m
    @field Z: the topic corpus, just same as W,
              except Z[m][n] is the id of the topic of the term
    @field M: the number of documents
    @field T: the number of terms
    @field WN: the number of all words in W
    @field LN: the number of all original labels
    @field iteration: the times of iteration
    @field Doc2TopicCount: a matrix, shape is M * K,
                           Doc2TopicCount[m][k] is the times of topic k sampled in document m
    @field Topic2TermCount: a matrix, shape is K * T,
                            Topic2TermCount[k][t] is the times of term t generated from topic k

    # derivative fields
    @field Doc2TopicCountSum: a vector, shape is M, self.Doc2TopicCount.sum(axis=1)
                              Doc2TopicCountSum[m] is the count of all topic,
                              i.e., Doc2TopicCountSum[m] is the number of words in document m
    @field alpha_vector_Lambda: a matrix, self.alpha_vector * self.Lambda
    @alpha_vector_Lambda_sum: a vector, self.alpha_vector_Lambda.sum(axis=1)
    @eta_vector_sum: float value, sum(self.eta_vector)
    @Topic2TermCountSum: a vector, self.Topic2TermCount.sum(axis=1)

    """
    def __init__(self, alpha_vector=None, eta_vector=None, labeled_documents=None):
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

        self.Doc2TopicCount = None
        self.Topic2TermCount = None
        self.Lambda = None

        # derivative fields:
        # the following fields could reduce operations in training and inference
        # it is not necessary to save them to file, we can recover them by other fields
        self.Doc2TopicCountSum = None
        self.alpha_vector_Lambda = None
        self.alpha_vector_Lambda_sum = None
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
        self.Doc2TopicCountSum = self.Doc2TopicCount.sum(axis=1)
        self.alpha_vector_Lambda = self.alpha_vector * self.Lambda
        self.alpha_vector_Lambda_sum = self.alpha_vector_Lambda.sum(axis=1)
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
            doc_words = document.split()
            # self.documents.append(doc_words)
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
            # label_counter = Counter(all_labels)
            # counter_list = [(self.topic_vocabulary[label], count) for label, count in label_counter.items()]
            # sorted_count_list = sorted(counter_list, key=lambda counter: counter[0], reverse=False)
            # print self.topics
            # print sorted_count_list
            # count_vector = np.array([count for _, count in sorted_count_list], dtype=float)
            # self.alpha_vector = count_vector / sum(count_vector)
            # print self.alpha_vector
            # print sorted_count_list
            # self.alpha_vector = [50.0/self.K for _ in range(self.K)]
            self.alpha_vector = [0.001 for _ in range(self.K)]
        if self.eta_vector is None:
            self.eta_vector = [0.001 for _ in range(self.T)]

        self.Z = []
        for m in range(self.M):
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

        self._initialize_derivative_fields()
        pass

    @staticmethod
    def _multinomial_sample(p_vector):
        """
        sample a number from multinomial distribution
        :param p_vector: the probabilities
        :return: a int value
        """
        return np.random.multinomial(1, p_vector).argmax()

    def _gibbs_sample_training(self):
        """
        sample a topic(k) for each word(t) of all documents, Generate a new matrix Z
        :return: None
        """
        # TODO: the operations of addition and multiplication could be reduced, because some
        count = 0
        for m in range(self.M):

            # doc_m_eta_vector = self.eta_vector
            # doc_m_alpha_vector = self.alpha_vector * self.Lambda[m]
            doc_m_alpha_vector = self.alpha_vector_Lambda[m]
            # assert (doc_m_alpha_vector == self.alpha_vector_Lambda[m]).all()

            # sum_doc_m_alpha_vector = sum(doc_m_alpha_vector)
            sum_doc_m_alpha_vector = self.alpha_vector_Lambda_sum[m]
            # assert sum_doc_m_alpha_vector == self.alpha_vector_Lambda_sum[m]

            for t, z, n in zip(self.W[m], self.Z[m], range(len(self.W[m]))):
                k = z
                self.Doc2TopicCount[m, k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = self.Doc2TopicCount[m] + doc_m_alpha_vector
                # denominator_theta = sum(self.Doc2TopicCount[m]) + sum_doc_m_alpha_vector
                denominator_theta = self.Doc2TopicCountSum[m]-1 + sum_doc_m_alpha_vector
                # assert sum(self.Doc2TopicCount[m]) == self.Doc2TopicCountSum[m]-1

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + sum(self.eta_vector)
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + self.eta_vector_sum
                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum
                # assert (self.Topic2TermCount.sum(axis=1) == self.Topic2TermCountSum).all()
                # assert sum(self.eta_vector) == self.eta_vector_sum

                beta_vector = 1.0 * numerator_beta_vector / denominator_beta
                theta_vector = 1.0 * numerator_theta_vector / denominator_theta

                p_vector = beta_vector * theta_vector
                # print p_vector
                p_vector = p_vector / sum(p_vector)
                # print p_vector
                sample_z = LldaModel._multinomial_sample(p_vector)
                self.Z[m][n] = sample_z

                k = sample_z
                self.Doc2TopicCount[m, k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1
                count += 1
        assert count == self.WN
        print "gibbs sample count: ", self.WN
        self.iteration += 1
        pass

    def _gibbs_sample_inference(self, term_vector, iteration, times=10):
        """
        inference with gibbs sampling
        :param term_vector: the term vector of document
        :param iteration: the times of iteration
        :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """
        doc_topic_count = np.zeros(self.K, dtype=int)
        p_vector = np.ones(self.K, dtype=int)
        p_vector = p_vector * 1.0 / sum(p_vector)
        z_vector = [LldaModel._multinomial_sample(p_vector) for _ in term_vector]
        for n, t in enumerate(term_vector):
            k = z_vector[n]
            doc_topic_count[k] += 1
            self.Topic2TermCount[k, t] += 1
            self.Topic2TermCountSum[k] += 1

        sum_doc_topic_count = sum(doc_topic_count)
        doc_m_alpha_vector = self.alpha_vector
        sum_doc_m_alpha_vector = sum(doc_m_alpha_vector)
        for i in range(iteration):
            for n, t in enumerate(term_vector):
                k = z_vector[n]
                doc_topic_count[k] -= 1
                self.Topic2TermCount[k, t] -= 1
                self.Topic2TermCountSum[k] -= 1

                numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
                denominator_theta = sum_doc_topic_count - 1 + sum_doc_m_alpha_vector

                numerator_beta_vector = self.Topic2TermCount[:, t] + self.eta_vector[t]
                # denominator_beta = self.Topic2TermCount.sum(axis=1) + sum(self.eta_vector)
                denominator_beta = self.Topic2TermCountSum + self.eta_vector_sum

                beta_vector = numerator_beta_vector / denominator_beta
                theta_vector = numerator_theta_vector / denominator_theta

                p_vector = beta_vector * theta_vector
                # print p_vector
                p_vector = p_vector / sum(p_vector)
                # print p_vector
                sample_z = LldaModel._multinomial_sample(p_vector)
                z_vector[n] = sample_z

                k = sample_z
                doc_topic_count[k] += 1
                self.Topic2TermCount[k, t] += 1
                self.Topic2TermCountSum[k] += 1
        # reset self.Topic2TermCount
        for n, t in enumerate(term_vector):
            k = z_vector[n]
            self.Topic2TermCount[k, t] -= 1
            self.Topic2TermCountSum[k] -= 1

        numerator_theta_vector = doc_topic_count + doc_m_alpha_vector
        denominator_theta = sum(doc_topic_count) + sum(doc_m_alpha_vector)
        theta_new = 1.0 * numerator_theta_vector / denominator_theta
        return theta_new

    def training(self, iteration=10, log=True):
        """
        training this model with gibbs sampling
        :param log: print perplexity after every gibbs sampling if True
        :param iteration: the times of iteration
        :return: None
        """
        for i in range(iteration):
            if log is True:
                print "after iteration: %s, perplexity: %s" % (i, self.perplexity)
            self._gibbs_sample_training()
        pass

    def training_one(self, iteration=10):
        """
        training this model with gibbs sampling only once
        :param iteration: the times of iteration
        :return: None
        """
        for i in range(iteration):
            print "after iteration: %s, perplexity: %s" % (i, self.perplexity)
            self._gibbs_sample_training()

    def inference(self, document, iteration=10, times=10):
        # TODO: inference of a document
        """
        inference for one document
        :param times: the times of gibbs sampling, the result is the average value of all times(gibbs sampling)
        :param iteration: the times of iteration
        :param document: some sentence like "this is a method for inference"
        :return: theta_new, a vector, theta_new[k] is the probability of doc(term_vector) to be generated from topic k
                 theta_new, a theta_vector, the doc-topic distribution
        """

        words = document.split()
        term_vector = [self.vocabulary[word] for word in words if word in self.vocabulary]
        theta_new_accumulation = np.zeros(self.K, float)
        for time in range(times):
            theta_new = self._gibbs_sample_inference(term_vector, iteration=iteration)
            # print theta_new
            theta_new_accumulation += theta_new
        theta_new = theta_new_accumulation / times
        # print "avg: \n", theta_new
        doc_topic_new = [(self.topics[k], probability) for k, probability in enumerate(theta_new)]
        sorted_doc_topic_new = sorted(doc_topic_new,
                                      key=lambda topic_probability: topic_probability[1],
                                      reverse=True)
        return sorted_doc_topic_new
        pass

    def beta_k(self, k):
        """
        topic-term distribution
        beta_k[t] is the probability of term t(word) to be generated from topic k
        :return: a vector, shape is T
        """
        numerator_vector = self.Topic2TermCount[k] + self.eta_vector
        denominator = sum(self.Topic2TermCount[k]) + sum(self.eta_vector)
        return 1.0 * numerator_vector / denominator

    def theta_m(self, m):
        """
        doc-topic distribution
        theta_m[k] is the probability of doc m to be generated from topic k
        :return: a vector, shape is K
        """
        numerator_vector = self.Doc2TopicCount[m] + self.alpha_vector * self.Lambda[m]
        denominator = sum(self.Doc2TopicCount[m]) + sum(self.alpha_vector * self.Lambda[m])
        return 1.0 * numerator_vector / denominator

    @property
    def beta(self):
        """
        topic-term distribution
        beta[k, t] is the probability of term t(word) to be generated from topic k
        :return: a matrix, shape is K * T
        """
        numerator_matrix = self.Topic2TermCount + self.eta_vector
        # column vector
        denominator_vector = self.Topic2TermCount.sum(axis=1).reshape(self.K, 1) + sum(self.eta_vector)
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

    @property
    def log_perplexity(self):
        """
        log perplexity of LDA topic model
        :return: a float value
        """
        beta = self.beta
        # theta = self.theta
        log_likelihood = 0
        word_count = 0
        for m, theta_m in enumerate(self.theta):
            for t in self.W[m]:
                likelihood_t = np.inner(beta[:, t], theta_m)
                # print likelihood_t
                log_likelihood += -np.log(likelihood_t)
                word_count += 1
        assert word_count == self.WN, "word_count: %s\tself.WN: %s" % (word_count, self.WN)
        return 1.0 * log_likelihood / self.WN

    @property
    def perplexity(self):
        """
        perplexity of LDA topic model
        :return: a float value, perplexity = exp{log_perplexity}
        """
        return np.exp(self.log_perplexity)

    def __repr__(self):
        return "Labeled-LDA Model:\n" \
               "\tK = %s\n" \
               "\tM = %s\n" \
               "\tT = %s\n" \
               "\tWN = %s\n" \
               "\tLN = %s\n" \
               "\tperplexity = %s\n" \
               "\t" % (self.K, self.M, self.T, self.WN, self.LN, self.perplexity)
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
            self.Doc2TopicCount = None
            self.Topic2TermCount = None
            self.Lambda = None

            if save_model_dict is not None:
                self.__dict__ = save_model_dict
        pass

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
                json.dump(target_object, f, skipkeys=False, ensure_ascii=False, check_circular=True, allow_nan=True, cls=None, indent=True, separators=None, encoding="utf-8", default=None, sort_keys=False)
        except Exception, e:
            message = "Write [%s...] to file [%s] error: json.dump error" % (str(target_object)[0:10], file_name)
            print ("%s\n\t%s" % (message, e.message))
            print "e.message: ", e.message
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

    def save_model_to_dir(self, dir_name):
        """
        save model to directory dir_name
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

        np.save(os.path.join(dir_name, "Doc2TopicCount.npy"), self.Doc2TopicCount)
        np.save(os.path.join(dir_name, "Topic2TermCount.npy"), self.Topic2TermCount)
        np.save(os.path.join(dir_name, "Lambda.npy"), self.Lambda)
        pass

    def load_model_from_dir(self, dir_name):
        """
        load model from directory dir_name
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

        self.Doc2TopicCount = np.load(os.path.join(dir_name, "Doc2TopicCount.npy"))
        self.Topic2TermCount = np.load(os.path.join(dir_name, "Topic2TermCount.npy"))
        self.Lambda = np.load(os.path.join(dir_name, "Lambda.npy"))

        self._initialize_derivative_fields()
        pass


if __name__ == "__main__":
    pass
