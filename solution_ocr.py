# -*- coding: utf-8 -*-

#####
# Nader Baydoun (20156885)
###


# Utiliser dbg() pour faire un break dans votre code.
# W is the matrice de poids so it contains the values between two nodes.
# W[i,j] represents the connection between the i hidden neurone and the j input neuron
# w[i] contains the weight between the i hidden neuron the output neurone
# X a 2D matrix of 2000 inputs, and each input has 128 values that need to be iterated over
# Y is an array of 2000 values, the classification result of every input
from pdb import set_trace as dbg
import numpy as np


def logistic(x):
    return 1. / (1. + np.exp(-x))


class ReseauDeNeurones:

    def __init__(self, alpha, T):
        self.alpha = alpha
        self.T = T

    """
    Initialise la matrice de poids W entre la couche d'entree et la
    couche cachee, puis le vector de connexions w entre la couche cachee et le neurone de sortie. W est
    donc un tableau Numpy a deux dimensions (matrice) et w est un tableau Numpy a une dimension (vector).
    Plus speciquement, la valeur de la connexion entre le ie neurone cache et la je entree xj correspond
    a W[i,j]. De plus, la connexion entre le ie neurone cache et le neurone de sortie correspond
    a w[i]. Cette methode est deja implementee.
    """
    def initialisation(self, W, w):
        self.W = W
        self.w = w

    """
    Retourne la paire (W,w) de la matrice de connexions W (c'est-a-dire une
    matrice Numpy W) et du vector de connexions w (c'est-a-dire un vector Numpy w) du reseau de
    neurones. Cette methode est deja implementee.
    """
    def parametres(self):
        return (self.W, self.w)

    """
    Retourne la prediction par le reseau de neurones de la classe d'une entree,
    representee par un vecteur Numpy x. Cette prediction doit donc etre 0 ou 1.
    """
    def prediction(self, x):
        hidden_layer = [0] * len(self.W)
        #Forward-prop
        for i in range(len(hidden_layer)):
            cumulative_sum = 0
            for j in range(len(self.W[i])):
                cumulative_sum += self.W[i, j] * x[j]
            hidden_layer[i] = logistic(cumulative_sum)

        cumulative = 0
        for i in range(len(hidden_layer)):
            cumulative += hidden_layer[i] * self.w[i]
        final_value = logistic(cumulative)
        return final_value


    """
    Met a jour les parametres du reseau de neurones a l'aide de sa regle
    d'apprentissage, a partir d'une entree x (vecteur Numpy) et de sa classe cible y (0 ou 1).
    """
    def mise_a_jour(self, x, y):
        hidden_layer = [0] * len(self.W)

        #Forward-prop
        for i in range(len(hidden_layer)):
            cumulative_sum = 0
            for j in range(len(self.W[i])):
                cumulative_sum += self.W[i, j] * x[j]
            hidden_layer[i] = logistic(cumulative_sum)

        cumulative = 0
        for i in range(len(hidden_layer)):
            cumulative += hidden_layer[i] * self.w[i]
        final_value = logistic(cumulative)

        #Back-prop
        final_delta = y - final_value
        hidden_layer_deltas = [0] * len(self.W)
        for i in range(len(hidden_layer_deltas)):
            hidden_layer_deltas[i] = hidden_layer[i] * (1 - hidden_layer[i]) * self.w[i] * y

        for iterator in range(len(self.w)):
            self.w[iterator] = self.w[iterator] + self.alpha * final_value * final_delta

        for j in range(len(x)):
            for i in range(len(hidden_layer)):
                self.W[i, j] = self.W[i, j] + self.alpha * hidden_layer[i] * hidden_layer_deltas[i]

    """
    Entraine le reseau de neurones durant T iterations sur l'ensemble
    d'entrainement forme des entrees X (une matrice Numpy, ou la t^e rangee correspond a l'entree xt) et
    des classes cibles Y (un vecteur Numpy ou le t^e element correspond a la cible yt). Il est recommande
    d'appeler votre methode mise a jour(self, x, y) a l'interieur de entrainement(self, X, Y).
    """
    def entrainement(self, X, Y):
        print(self.W)
        print("--------------------------")
        for iteration in range(self.T):
            for input_value in range(len(X)):
                self.mise_a_jour(X[input_value],Y[input_value])
                #lol = 1
        print(self.W)


