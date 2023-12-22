
class Prediction:
    def __init__(self, surrogate, method):
        self.surrogate = surrogate
        self.method = method

    def predict(self, composition, cell, topk=1):
        """
        Predicts the structure of a composition

        Args:
            composition (str): A string representing a chemical composition
            topk (int, optional): Number of minima to return. Defaults to 1.

        Returns:
            list: A list of ase.Atoms objects representing the predicted minima
        """
        minima = self.method.optimize(composition, cell, topk)
        return minima
    