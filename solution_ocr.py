# -*- coding: utf-8 -*-

#####
# Nader Baydoun (20156885)
###


# Utiliser dbg() pour faire un break dans votre code.
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
        #TODO: .~= À COMPLÉTER =~.
        hidden_layer = [0] * len(self.W)

        #Forward propagation
        for input_value in range(len(x)):
            # Populate the elements of our network
            for i in range(len(hidden_layer)):
                cumulative_sum = 0
                for j in range(len(self.W[i])):
                    cumulative_sum += self.W[i, j] * x[input_value]
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
        #TODO: .~= À COMPLÉTER =~.
        # entrainement function does the looping and mise_a_jour updates the values
        pass

    """
    Entraine le reseau de neurones durant T iterations sur l'ensemble
    d'entrainement forme des entrees X (une matrice Numpy, ou la t^e rangee correspond a l'entree xt) et
    des classes cibles Y (un vecteur Numpy ou le t^e element correspond a la cible yt). Il est recommande
    d'appeler votre methode mise a jour(self, x, y) a l'interieur de entrainement(self, X, Y).
    """
    def entrainement(self, X, Y):
        #TODO: .~= À COMPLÉTER =~.

        #TRUTH:
        # W is the matrice de poids so it contains the values between two nodes.
        # W[i,j] represents the connection between the i hidden neurone and the j input neuron
        # w[i] contains the weight between the i hidden neuron the output neurone
        # X a 2D matrix of 2000 inputs, and each input has 128 values that need to be iterated over
        # Y is an array of 200 values, the classification result of every input

        #Build the neurones that represent the hidden layer of our net
        #Initialise a list that will be our hidden layer
        hidden_layer = [0] * len(self.W)

        for iteration in range(self.T):
            for input_value in range(len(X)):
                # Populate the elements of our network
                for i in range(len(hidden_layer)):
                    cumulative_sum = 0
                    # Calculate the value of the node, you will need the logistic function, but first you need the summation of:
                    # input_node_value*edge_weight
                    for j in range(len(self.W[i])):
                        cumulative_sum += self.W[i, j] * X[input_value][j]
                    hidden_layer[i] = logistic(cumulative_sum)

                # Calculate final value of our last output node
                cumulative = 0
                for i in range(len(hidden_layer)):
                    cumulative += hidden_layer[i] * self.w[i]
                final_value = logistic(cumulative)


                final_delta = Y[input_value] - final_value

                hidden_layer_deltas = [0] * len(self.W)

                for i in range(len(hidden_layer_deltas)):
                    hidden_layer_deltas[i] = hidden_layer[i] * (1 - hidden_layer[i]) * self.w[i] * Y[i]

                #UNRESOLVED ISSUE: We have to compare to Y before back-propagating? Why? is this not the right value?
                #Update every weight in network using deltas
                #for each weight w[i,j] in network do
                #   w[i,j] = w[i,j] + alpha * a[i] * delta[j]
                #   weight = weight + self.alpha * value_of_node * delta of node

                # We need to update the weight from hidden to output - easy
                #TODO: We might not need this?
                for outputer in range(len(self.w)):
                    self.w[outputer] = self.w[outputer] + self.alpha * final_value * final_delta # This is the Y

                # Then we need to update the weight from input to hidden - need two loops
                # W[i,j] represents the connection between the i hidden neurone and the j input neuron
                for j in range(len(X[input_value])):
                    for i in range(len(hidden_layer)):
                        self.W[i,j] = self.W[i,j] + self.alpha * hidden_layer[i] * hidden_layer_deltas[i] #TODO: Maybe move to mise a jour?
