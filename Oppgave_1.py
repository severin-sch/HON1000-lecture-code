import numpy as np
import matplotlib.pyplot as plt

"""
Oppgave: finn vekter og bias til et perceptron med 3 inputs og en output.

Instruksjoner: 
1. 
Del opp klassen i grupper og be dem finne vekter og 
bias for å predikere på følgende datasett:

x_train = np.array([[1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                    [0, 1, 0]])

y_train = np.array([0, 1, 0, 1])

2. 
Kjør programmet og skriv inn parametrene til hver av gruppene i prompten. 
Hver modell blir så testet på treningssett og testsett. Til slutt får 
dere en graf til å sammenligne gruppene.

"""


class Perceptron:
    def __init__(self, w0, w1, w2, b):
        self.w = np.array([w0, w1, w2])
        self.b = b

    def predict(self, x0, x1, x2):
        w, b = self.w, self.b
        x = np.array([x0, x1, x2])
        a = (x.T @ w) + b
        if a > 0:
            return 1
        else:
            return 0

if __name__ == "__main__":
    print("How many models are you making? Integers please.")
    n = int(input("> "))
    models = []
    for i in range(n):
        print(f"Weights and biases for model {i}, please.")
        print("Format: w0, w1, w2, b")
        p = (input("> ")).split(",")
        models.append(Perceptron(float(p[0]), float(p[1]), float(p[2]), float(p[3])))

    x_train = np.array([[1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1],
                        [0, 1, 0]])

    y_train = np.array([0, 1, 0, 1])

    x_test = np.array([[0, 0, 0],
                       [1, 1, 0],
                       [1, 0, 0],
                       [0, 0, 1]])

    y_test = np.array([0, 1, 0, 0])

    print("Finding error rate in training set...")

    training_scores = np.zeros(n)
    for i, model in enumerate(models):
        modelscore = 0
        for x,y in zip(x_train,y_train):
            a = model.predict(x[0], x[1], x[2])
            if a == y:
                modelscore += 1
        training_scores[i] = modelscore

    print("Finding error rate in testing set...")
    testing_scores = np.zeros(n)
    for i, model in enumerate(models):
        modelscore = 0
        for x, y in zip(x_test, y_test):
            a = model.predict(x[0], x[1], x[2])
            if a == y:
                modelscore += 1
        testing_scores[i] = modelscore

    print("Completed testing")

    testing_scores *= 25
    training_scores *= 25


    lablocs = np.arange(len(testing_scores))
    width = 0.25
    labels = []
    for i in range(len(testing_scores)):
        labels.append("Model " + str(i))

    fig, ax = plt.subplots()
    rects1 = ax.bar(lablocs - width / 2, testing_scores, width, label='Test score')
    rects2 = ax.bar(lablocs + width / 2, training_scores, width, label='Training score')

    modelnumber = list(range(len(testing_scores)))

    ax.set_ylabel('% correct predictions')
    ax.set_title('Model scores')
    ax.set_xticks(lablocs)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.ylim(0, 105)
    fig.tight_layout()
    print("Showing results")
    plt.show()
