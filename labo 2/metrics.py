import numpy as np
    
def edit_distance(a,b):
    # Calcul de la distance d'édition

    # ---------------------- Laboratoire 2 - Question 1 - Début de la section à compléter ------------------
    dp = np.arange(len(b) + 1)
    for i, char_a in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, char_b in enumerate(b, 1):
            insert, delete, replace = dp[j] + 1, dp[j - 1] + 1, prev + (char_a != char_b)
            prev, dp[j] = dp[j], min(insert, delete, replace)
    return dp[-1]
    # ---------------------- Laboratoire 2 - Question 1 - Fin de la section à compléter ------------------

if __name__ =="__main__":
    a = list('allo')
    b = list('apollo2')
    c = edit_distance(a,b)

    print('Distance d\'edition entre ',''.join(a),' et ',''.join(b), ': ', c)
    