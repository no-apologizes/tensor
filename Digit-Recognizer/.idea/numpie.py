class NumPyArray:
    def __init__(self, andie):
        self.andie = andie

    def __truediv__(self, other):
        # Intercepts scaling via () / {other}
        return self