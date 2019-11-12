import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from .dish_class import DiSH

        data = np.array([
                         [1, 1],
                         [0, 0],
                         [0, 2],
                         [2, 0],
                         [2, 2],
                         ])

        algo = DiSH(epsilon=1, mu=1)
        algo.data = data
        algo._get_neighbors(data[0], features=[0])

        plt.plot(data[:,0], data[:,1], 'o')
        plt.plot(data[:,0], data[:,1]*0, 'go')
        plt.plot(data[:,0], data[:,1]*0, 'go')


        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
