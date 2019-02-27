import csv
import random
import nltk


def get_labeled_reviews(path_to_csv):
    labeled_reviews = []
    with open(path_to_csv, newline='', encoding='utf-8') as csvfile:
        review_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(review_reader, None)  # Skip csv headers
        for row in review_reader:
            label = int(row[0])
            review_text = row[2]

            review = (review_text, label)
            labeled_reviews.append(review)

    return labeled_reviews


def review_features(review, all_words):
    #features = {}
    features = all_words.copy()
    #features["review"] = review
    for word in str.split(review, " "):
        if len(word) > 1:
            if word in features:
                features[word] += 1
            else:
                features[word] = 1
    return features

def cross_validation(all_data, n_sets):
    set_size = 1.0 / n_sets
    shuffled_data = all_data.copy()
    random.shuffle(shuffled_data)
    cumulative_percent = 0
    for i in range(0, 2):
        n_training = int(set_size * len(all_data))
        split_start = i * n_training
        split_end = (i + 1) * n_training
        print("train split_start: " + str(split_start) + " - split_end: " + str(split_end))
        train_data_before = shuffled_data[:split_start]
        train_data_after = shuffled_data[split_end:]
        train_data = train_data_before + train_data_after
        test_data = shuffled_data[split_start:split_end]
        print('{}\n{}\n{}'.format(train_data_before, train_data_after, train_data))
        # print("train size: " + str(len(train_data)) + " - test size: " + str(len(test_data)))
        classifier = nltk.NaiveBayesClassifier.train(train_data, nltk.LaplaceProbDist)
        correct = 0
        for i, (t, l) in enumerate(test_data):
            classified = classifier.classify(t)
            # actual = labeled_reviews[split_point:][i][1]
            if classified == l:
                correct += 1
        print(str(correct) + "/" + str(len(test_data)))
        correct_percent = correct/len(test_data)
        cumulative_percent += correct_percent
        print(str(correct_percent) + "%")
    print("Average result: " + str(cumulative_percent / n_sets) + "%")







