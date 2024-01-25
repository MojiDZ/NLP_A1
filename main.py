import numpy as np
from classifier import Classifier


def main():
    classifier = Classifier('./corpus/training')
    s_labels, g_labels = classifier.classify('./corpus/testing')

    # confusion Matrix
    confusion_matrix = np.zeros([2, 2])
    for label in range(len(s_labels)):
        confusion_matrix[int(s_labels[label])][int(g_labels[label])] += 1

    tp, fp, fn, tn = confusion_matrix.flatten()

    # Measurements of performance
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_1 = (2 * precision * recall) / (precision + recall)

    print("Confusion Matrix:\n", confusion_matrix)
    print('Accuracy: ' + str(accuracy))
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f_1))


if __name__ == '__main__':
    main()
