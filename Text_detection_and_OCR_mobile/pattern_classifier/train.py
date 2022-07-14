import numpy as np
from sklearn import neural_network, tree
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, \
    make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

from price_detector.data_processing import read_pickle_local, write_pickle_local

random_state = 42


def train(features_file: str, labels_file: str, params, test_size: float = .15, n_jobs=3,
          cv=3, number_of_boxes_tuple=(2, 3, 4)):
    features = read_pickle_local(features_file)
    labels = read_pickle_local(labels_file)
    # num_boxes = 4
    scoring = make_scorer(f1_score, True, average='micro')
    for num_boxes in number_of_boxes_tuple:
        X = features[num_boxes]
        y_true = labels[num_boxes]
        param_grid = params[num_boxes]
        X_train, X_val, y_train, y_val = train_test_split(X, y_true,
                                                          test_size=test_size,
                                                          random_state=random_state)
        print(np.array(X_train).shape, np.array(X_val).shape)
        tree_classifier = tree.DecisionTreeClassifier(class_weight='balanced',
                                                      random_state=random_state)
        clf = GridSearchCV(tree_classifier, param_grid, scoring, n_jobs=n_jobs, cv=cv)
        clf.fit(X_train, y_train)
        print(param_grid, '\n', clf.best_params_)
        y_pred_val = clf.predict(X_val)
        y_pred_train = clf.predict(X_train)
        print("train\n", classification_report(y_train, y_pred_train))
        print("balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_train,
                                                                        y_pred_train))
        print("val\n", classification_report(y_val, y_pred_val))
        print(
            "balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_val, y_pred_val))

def create_model(features_size, units1, units2, lr, n_classes):
    import tensorflow as tf
    model = tf.keras.Sequential()
    input_shape = (features_size,)
    model.append(tf.keras.layers.Dense(units1, input_shape=input_shape, activation='relu'))
    model.append(tf.keras.layers.Dense(units2, activation='softmax'))
    model.append(tf.keras.layers.Dense(n_classes, activation='softmax'))
    optimizer = tf.optimizers.Adam(lr)
    model.compile(optimizer, tf.keras.losses.categorical_crossentropy)
    # TODO keras utils to categorical - сработает, нужен маппинг классов
    return model

def save_nn(features_file: str, labels_file: str, save_file: str, test_size: float = .15):
    features = read_pickle_local(features_file)
    labels = read_pickle_local(labels_file)
    # num_boxes = 4
    max_iter = 250
    # max_iter = 600
    max_iter2 = 350
    # max_iter2 = 1000
    max_iters = {2: max_iter, 3: max_iter, 4: max_iter, 5: max_iter2, 6: max_iter2}
    learning_rate = .001
    res = {}
    for num_boxes in max_iters.keys():
        X = features[num_boxes]
        y_true = labels[num_boxes]
        X_train, X_val, y_train, y_val = train_test_split(X, y_true, test_size=test_size,
                                                          random_state=random_state)
        print(np.array(X_train).shape, np.array(X_val).shape)

        max_iter = max_iters[num_boxes]
        hidden_layer_sizes = (num_boxes * 8,  num_boxes * 8)
        classifier = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                                  max_iter=max_iter,
                                                  learning_rate_init=learning_rate)
        # features_size = num_boxes * 4
        # classifier = create_model(features_size, features_size * 2, features_size * 2,
        #                           learning_rate)
        classifier.fit(X_train, y_train)
        y_pred_val = classifier.predict(X_val)
        y_pred_train = classifier.predict(X_train)
        print("train")
        print(classification_report(y_train, y_pred_train))
        print("balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_train,
                                                                        y_pred_train))
        print("val")
        print(classification_report(y_val, y_pred_val))
        print(
            "balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_val, y_pred_val))
        res[num_boxes] = classifier
    write_pickle_local(save_file, res)


def save_trees(features_file: str, labels_file: str,
               params, save_file: str, test_size: float = .15,
               number_of_boxes_tuple=(2, 3, 4, 5)):
    features = read_pickle_local(features_file)
    labels = read_pickle_local(labels_file)
    # num_boxes = 4
    max_iters = {2: 250, 3: 250, 4: 250, 5: 300}
    res = {}
    for num_boxes in number_of_boxes_tuple:
        X = features[num_boxes]
        y_true = labels[num_boxes]
        param_grid = params[num_boxes]
        X_train, X_val, y_train, y_val = train_test_split(X, y_true, test_size=test_size,
                                                          random_state=random_state)
        print(np.array(X_train).shape, np.array(X_val).shape)

        # tree_classifier = tree.DecisionTreeClassifier(class_weight='balanced',
        #                                               random_state=random_state,
        #                                               **param_grid)
        # tree_classifier.fit(X_train, y_train)
        # y_pred_val = tree_classifier.predict(X_val)
        # y_pred_train = tree_classifier.predict(X_train)
        max_iter = max_iters[num_boxes]
        classifier = neural_network.MLPClassifier(hidden_layer_sizes=(30,),
                                                  max_iter=max_iter)
        classifier.fit(X_train, y_train)
        y_pred_val = classifier.predict(X_val)
        y_pred_train = classifier.predict(X_train)
        print("train")
        print(classification_report(y_train, y_pred_train))
        print("balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_train,
                                                                        y_pred_train))
        print("val")
        print(classification_report(y_val, y_pred_val))
        print(
            "balanced_accuracy_score: %.3f" % balanced_accuracy_score(y_val, y_pred_val))
        res[num_boxes] = classifier
    write_pickle_local(save_file, res)
