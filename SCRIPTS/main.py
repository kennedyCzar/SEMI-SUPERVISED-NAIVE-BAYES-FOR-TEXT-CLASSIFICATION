#import requirements_elicitation
#import topic_grabber
#import sentiment
#import chen_preprocessing
import naive_bayes
#import em
import nltk

#
# reviews = requirements_elicitation.read_reviews("C:\\Users\\Matthew\\Downloads\\regdatapreprocessing\\my-tracks-reviews.csv")
#
# total = 0
# for r in reviews:
#     total = total + int(r.rating)
#
# avg = total / len(reviews)
#
# print("Average rating: " + str(avg))

# reviews = requirements_elicitation.read_reviews("C:\\Users\\Matthew\\Downloads\\mytracks_dataset.csv")
# tg = topic_grabber.TopicGrabber(reviews)
#
# print("Total reviews: " + str(len(tg.reviews)))
# # filtered = tg.filter_no_nouns()
# # for f in filtered:
# #     print(f.comment)
# print("No nouns: " + str(len(tg.filter_no_nouns())))
# #print("Bad nouns: " + str(len(tg.filter_blacklist_nouns())))
# filtered = tg.filter_blacklist_nouns()
# print("Final count: " + str(len(tg.reviews)))
#
# sentiment.measure_sentiment(tg.reviews)
#
# for r in tg.reviews:
#     print(r.comment)
#     print("compound: {0}, neg: {1}, neu: {2}, pos: {3}".format(r.sentiment['compound'], r.sentiment['neg'],
#                                                                r.sentiment['neu'], r.sentiment['pos']))

#reviews_split = chen_preprocessing.read_reviews("C:\\Users\\Matthew\\Downloads\\my-tracks-reviews.csv", True)
#reviews_unified = chen_preprocessing.read_reviews("C:\\Users\\Matthew\\Downloads\\my-tracks-reviews.csv", False)
#chen_preprocessing.export(reviews_split, "C:\\Users\\Matthew\\Downloads\\mytracks_exported_split.csv")
#chen_preprocessing.export(reviews_unified, "C:\\Users\\Matthew\\Downloads\\mytracks_exported_unified.csv")

### Working NB WITHOUT EM

labeled_reviews = naive_bayes.get_labeled_reviews("D:\\FREELANCER\\SEMI_NB_TEXT_CLASSIFICATION\\DATASET\\mytracks_NaiveBayes_Filter.csv")

all_words = {}
for (r, label) in labeled_reviews:
    for word in r.split(" "):
        if len(word) > 1:
            all_words[word] = 0
# print(len(all_words))

featuresets = [(naive_bayes.review_features(r, all_words), label) for (r, label) in labeled_reviews]

naive_bayes.cross_validation(featuresets, 10)

### END OF Working NB WITHOUT EM

### BEGIN 2nd Attempted EM Algorithm

# n_training = 3000
# labeled_data = featuresets[:n_training]
# unlabeled_data = featuresets[n_training:]
#
# classifier = nltk.NaiveBayesClassifier.train(labeled_data, nltk.LaplaceProbDist)
#
# max_iterations = 100
# for iteration in range(0, max_iterations):
#     print("Iteration: " + str(iteration))
#     found_labeled_data = []
#     correct = 0  # For evaluation not algorithm
#     for i, (t, l) in enumerate(unlabeled_data):
#         classified = classifier.classify(t)
#         if classified == l:  # For evaluation not algorithm
#             correct += 1  # For evaluation not algorithm
#         found_labeled_data.append((t, classified))
#     print(str(correct) + "/" + str(len(unlabeled_data)))
#     correct_percent = 100 * correct / len(unlabeled_data)
#     print(str(correct_percent) + "%")
#     classifier = nltk.NaiveBayesClassifier.train(labeled_data + found_labeled_data, nltk.LaplaceProbDist)

### END 2nd Attempted EM Algorithm

### BEGIN Attempted EM Algorithm

# labeled_reviews = naive_bayes.get_labeled_reviews("C:\\Users\\Matthew\\Downloads\\mytracks_NaiveBayes_Filter.csv")
# data_split = len(labeled_reviews)//5
# training_data = labeled_reviews[:data_split]
# unlabeled_data = labeled_reviews[data_split:]
#
# # Dictionary with counts of all words seen in training reviews
# all_words = {}
# for (r, label) in labeled_reviews:
#     for word in r.split(" "):
#         if len(word) > 1:
#             all_words[word] = 0
#
# # Train classifier on first 20% of reviews
# labeled_with_features = [(naive_bayes.review_features(r, all_words), label) for (r, label) in training_data]
# classifier = nltk.NaiveBayesClassifier.train(labeled_with_features)
#
# unlabeled_features = [naive_bayes.review_features(r, all_words) for (r, _) in unlabeled_data]
#
# max_iterations = 100
# log_lh_diff_min = 0.1
# log_lh_old = "first"
# for iteration in range(1, max_iterations):
#     print("Iteration: " + str(iteration))
#     # E Step
#     unlabeled_data = [(f, classifier.prob_classify(f)) for f in unlabeled_features]
#
#     # M Step
#     l_freqdist_act, ft_freqdist_act, ft_values_act = em.gen_freqdists(labeled_with_features, unlabeled_data)
#     l_probdist_act, ft_probdist_act = em.gen_probdists(l_freqdist_act, ft_freqdist_act, ft_values_act)
#     classifier = nltk.NaiveBayesClassifier(l_probdist_act, ft_probdist_act)
#
#     log_lh = sum([-classifier.prob_classify(ftdic).logprob(label) for (ftdic, label) in labeled_with_features])
#     log_lh += sum([-classifier.prob_classify(ftdic).logprob(label) for (ftdic, _) in unlabeled_data for label in
#                    l_freqdist_act])
#
#     # Continue until convergence
#     if log_lh_old == "first":
#         log_lh_old = log_lh
#     else:
#         log_lh_diff = log_lh - log_lh_old
#         print("log_lh_diff: " + str(log_lh_diff))
#         if log_lh_diff < log_lh_diff_min:
#             break
#         log_lh_old = log_lh
#
# print("EM Done, testing...")
# correct = 0
# for i, (t, l) in enumerate(unlabeled_data):
#     classified = classifier.classify(t)
#     # actual = labeled_reviews[split_point:][i][1]
#     if classified == l:
#         correct += 1
# print(str(correct) + "/" + str(len(unlabeled_data)))
# correct_percent = correct/len(unlabeled_data)
# print(str(correct_percent) + "%")

### END Attempted EM Algorithm

#split_point = 50
#train_set, test_set = featuresets[:split_point], featuresets[split_point:]
#classifier = nltk.NaiveBayesClassifier.train(train_set)

#total = len(test_set)
# print(test_set)
#correct = 0
#for i, (t, l) in enumerate(test_set):
#    classified = classifier.classify(t)
#    actual = labeled_reviews[split_point:][i][1]
#    if classified == actual:
#        correct += 1
#print(str(correct) + "/" + str(total))
#print(str(correct/total) + "%")

# print("Review: " + labeled_reviews[2000:][4][0] + " actual: " + str(labeled_reviews[2000:][4][1]) + " - classified ...")
# print(classifier.classify(test_set[4][0]))

#for t in tg.get_topics():
#    print(t)
#print (tg.get_topics())
