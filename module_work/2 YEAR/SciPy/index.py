from scipy import optimize
import numpy as np
from numpy.random import rand
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance_matrix

points = [(12, 1), (7, 17), (7, 3), (10, 12), (14, 15), (1, 2), (13, 11), (17, 16), (5, 18),
(10, 16), (9, 12), (3, 14), (12, 5), (19, 6), (13, 17), (17, 17), (1, 7), (1, 18), (19, 11), (14, 11),
(15, 11), (10, 1), (9, 2), (17, 12), (11, 18), (5, 5), (10, 0), (16, 0), (14, 1), (7, 9), (5, 3), (11,
13), (2, 2), (0, 19), (3, 12), (7, 8), (11, 12), (3, 9), (10, 2), (16, 1), (13, 18), (19, 5), (15, 6),
(17, 8), (8, 6), (10, 19), (19, 10), (16, 16), (11, 0), (6, 12)]

pointsCoolect = []
for item in points:
    pointsCoolect.append((list(item)))
points = np.array(pointsCoolect)

tri = Delaunay(pointsCoolect)
# plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()
hull = ConvexHull(points)
# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# plt.show()

vor = Voronoi(points)
# fig = voronoi_plot_2d(vor)
# plt.show()

points_1 = [(0, 0), (28, 13), (21, 24), (5, 17), (13, 8)]
points_1 = [list(item) for item in points_1 ]

points_2 = [(2, 17), (6, 6), (8, 25), (13, 28), (19, 15)]
points_2 = [list(item) for item in points_2 ]
sd = distance_matrix(np.array(points_1), np.array(points_2), p=5)
print(sd)



# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
#
# hull = ConvexHull(points)
# tri = Delaunay(points)
#
# vor = Voronoi(points)
# fig = voronoi_plot_2d(vor)
# plt.show()


# row = np.array([0, 2, 2, 0, 1, 2])
# col = np.array([0, 0, 1, 2, 2, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# sample_csc = sparse.csc_matrix((data, (row, col)), shape=(3, 3))
#
# row = np.array([0, 2, 2, 0, 1, 2])
# col = np.array([0, 0, 1, 2, 2, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
# csc = sparse.csc_matrix((data, (row, col)), shape=(3, 3))
#
# csr = csc.tocsr()
