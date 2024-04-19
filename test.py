# this is only used while creating to test snippets

"""id, survived, a, b, c
1,0,3,22,1,0
2,1,1,38,1,0
3,1,3,26,0,0
4,1,1,35,1,0
5,0,3,35,0,0
"""

from collections import Counter

my_list = [1, 2, 3, 4, 1, 2, 1, 1, 5]

element_counts = Counter(my_list)

print(element_counts[3])
