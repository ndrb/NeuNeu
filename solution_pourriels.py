# -*- coding: utf-8 -*-

#####
# Nader Baydoun (20156885)
###

from pdb import set_trace as dbg  # Utiliser dbg() pour faire un break dans votre code.

from collections import defaultdict

import numpy as np
import re
import math


# Probabilite: Classe permettant de modéliser les distributions P(C) et P(W|C).
#              Pour ce faire, les dictionnaires 'nbMotsParClasse', 'nbDocsParClasse',
#              'freqWC' doivent être remplis lors de la phase d'entraînement. La variable
#              membre vocabulaire sera automatique affectée après l'appel de la fonction
#              creerVocabulaire, vous n'avez donc pas à la modifiée.
#
#              Au final, lors de la prédiction, P, un objet de la classe 'Probabilite',
#              peut être appelé directement de cette façon : P(C=0) ou bien
#              P(W='allo',C=0,delta=1).
#
class Probabilite():

    def __init__(self):
        # Nb. de mots total dans les documents de la catégorie c.
        self.nbMotsParClasse = defaultdict(lambda: 0.)

        # Nb. de documents de la catégorie c.
        self.nbDocsParClasse = defaultdict(lambda: 0.)

        # Nb. de fois que le mot w apparaît dans les documents de la catégorie c.
        self.freqWC = defaultdict(lambda: 0.)

        # Vocabulaire des mots contenus dans tous les documents.
        self.vocabulaire = []

    """
    Calcule la probabilite a priori d'une classe. Par example, soit P un object
    de classe Probabilite, P.probClasses(C=0) retournera l'estimation de P(C = 0) (i.e. la
    probabilite a priori qu'un courriel soit un pourriel).
    P(C=c) = (# de document de categorie c) / (# de documents total) = |{t | ct = c}| / T
    """
    def probClasse(self, C):
        # P(C=c) = (# de document de categorie c) / (# de documents total)
        total_docs = self.nbDocsParClasse[0] + self.nbDocsParClasse[1]
        proba_class = self.nbDocsParClasse[C] / total_docs
        return proba_class



    """
    P(Wi=w|C=c) = # de fois que w apparait dans tous les documents de la categorie c / # de mots total dans les docs de cat c
    """
    def probMotEtantDonneClasse(self, C, W, delta):
        proba_mot_class = (delta+(self.freqWC[(W,C)])) / ((len(self.vocabulaire)+1)*delta + self.nbMotsParClasse[C])
        return proba_mot_class

    def __call__(self, C, W=None, delta=None):
        if W is None:
            return self.probClasse(C)

        return self.probMotEtantDonneClasse(C, W, delta)


# creerVocabulaire: Fonction qui s'occupe de créer une liste (i.e. un vocabulaire)
#                   des mots fréquents dans le corpus. Un mot est fréquent s'il
#                   apparaît au moins 'seuil' fois.
#
# documents: Liste de string représentant chacune le contenu d'un courriel.
#
# seuil: Fréquence minimale d'un mot pour qu'il puisse faire parti du vocabulaire.
#
# retour: Un 'set' contenant l'ensemble des mots ('str') du vocabulaire.
#
def creerVocabulaire(documents, seuil):
    vocab = set()
    dict_vocab = dict() #dict_vocab[key] = value

    for j in range(len(documents)):
        # Now I want to standardize my words in docs_word_list

        # You can remove all capitals
        #lower_doc = documents[j].lower()
        docs_word_list = documents[j].split() #same as using split(' ')

        #You can remove stop words (déterminant, pronom, vers commun comme être, avoir, faire)
        #Lemmatize the words so everything becomes infinitif

        # I want to iterate over docs_word_list and count the words I see while adding to dict_vocab
        for i in range(len(docs_word_list)):
            word = docs_word_list[i]
            if word in dict_vocab:
                cu_count = dict_vocab[word]
                cu_count += 1
                dict_vocab[word] = cu_count
            else:
                dict_vocab[word] = 1

    # Iterate over dict_vocab
    for k,v in dict_vocab.items():
        if v >= seuil:
            vocab.add(k)


    return vocab


# pretraiter: Fonction qui remplace les mots qui ne font pas parti du vocabulaire
#             par le token 'OOV'.
#
# doc: Un document représenté sous la forme d'une string.
#
# V: Vocabulaire représenté par un 'set' de mots ('str').
#
# retour: Une 'list' des mots contenu dans le document et présent dans le vocabulaire.
#
def pretraiter(doc, V):
    words_list = doc.split()
    new_doc = []
    for i in range(len(words_list)):
        word = words_list[i]
        if word in V:
            new_doc.append(word)
        else:
            new_doc.append("OOV")

    return new_doc


# entrainer: Fonction permettant d'entraîner les distributions P(C) et P(W|C)
#            à partir d'un ensemble de courriels.
#
# corpus: Liste de tuples, où chaque tuple est composé d'une liste de
#         mots (i.e. document prétraité) et un entier indiquant la
#         classe (0:SPAM,1:HAM). Par exemple,
#         corpus == [..., (["Mon", "courriel", "..."], 1), ...]
#
# P: Objet de la classe Probabilite qui doit être modifié directement (Référence!)
#
# retour: Rien! L'objet P doit être modifié via ses dictionnaires.
#
def entrainer(corpus, P):
    # Fill in nbMotsParClasse: Nb. de mots total dans les documents de la catégorie c.
    nbMotsParClasseZero = 0
    nbMotsParClasseUn = 0

    # Fill in nbDocsParClasse, Nb. de documents de la catégorie c.
    nbDocsParClasseZero = 0
    nbDocsParClasseUn = 0

    # Fill in freqWCNb. de fois que le mot w apparaît dans les documents de la catégorie c.
    # freqWC[('allo', 0)] donne le nombre de fois que allo apparait dans les docs de cat 0
    freqWCNbZero = dict()
    freqWCNbUn = dict()

    # i == (["Mon", "courriel", "..."], 1)
    for i in range(len(corpus)):
        if corpus[i][1] == 0:
            nbMotsParClasseZero += len(corpus[i][0])
            nbDocsParClasseZero += 1
            for j in range(len(corpus[i][0])):
                word = corpus[i][0][j]
                if word in freqWCNbZero:
                    counter = freqWCNbZero[word]
                    counter += 1
                    freqWCNbZero[word] = counter
                else:
                    freqWCNbZero[word] = 1
                    freqWCNbZero[word] = 1
        else:
            nbMotsParClasseUn += len(corpus[i][0])
            nbDocsParClasseUn += 1
            for j in range(len(corpus[i][0])):
                word = corpus[i][0][j]
                if word in freqWCNbUn:
                    counter = freqWCNbUn[word]
                    counter += 1
                    freqWCNbUn[word] = counter
                else:
                    freqWCNbUn[word] = 1
                    freqWCNbUn[word] = 1

    #Populate the values of P
    #k,v where k is the class number and v is the value
    P.nbMotsParClasse[0] = nbMotsParClasseZero
    P.nbMotsParClasse[1] = nbMotsParClasseUn

    P.nbDocsParClasse[0] = nbDocsParClasseZero
    P.nbDocsParClasse[1] = nbDocsParClasseUn

    for k,v in freqWCNbZero.items():
        P.freqWC[(k,0)] = v

    for k,v in freqWCNbUn.items():
        P.freqWC[(k, 1)] = v


# predire: Fonction utilisée pour trouver la classe la plus probable à quelle
#          appartient le document d à partir des distribution P(C) et P(W|C).
#
# doc: Un document représenté sous la forme d'une 'list' de mots ('str').
#
# P: Objet de la classe Probabilite.
#
# C: Liste des classes possibles (0:SPAM,1:HAM)
#
# delta: Paramètre utilisé pour le lissage.
#
# retour: Un tuple (int,float) où l'entier désigne la classe la plus probable
#         du document et le nombre à virgule est la log-probabilité conjointe
#         d'un document D=[w_1,...,w_d] et de catégorie c, i.e. P(C=c,D=[w_1,...,w_d]). *N'oubliez pas vos logarithmes!
#
def predire(doc, P, C, delta):
    #Start by calculating the porba posteriori pour chaque class, on commence par ajouter P(C=c)
    proba_posteriori_zero = math.log(P.probClasse(C[0]))
    proba_posteriori_un = math.log(P.probClasse(C[1]))

    #Sum of log P(Wi=wi | C=c)
    for i in range(len(doc)):
        word = doc[i]
        proba_posteriori_zero += math.log(P(word, 0, delta))
        proba_posteriori_un += math.log(P(word, 1, delta))

    #Then take the max argument and return
    if proba_posteriori_zero >= proba_posteriori_un:
        return 0, proba_posteriori_zero
    else:
        return 1, proba_posteriori_un
