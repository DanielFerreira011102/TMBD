import numpy as np

class IQROutlierDetector:
    def __init__(self, stream, alpha=1.9):
        self.stream = stream
        self.alpha = alpha
        self.results = self.detect_lower_outliers()
        self.print_results()

    def detect_lower_outliers(self):
        """
        Detect lower outliers using the IQR method with Q0.25 and Q0.75.
        Uses the formula: Q0.25 - alpha * (Q0.75 - Q0.25) as the lower bound.
        """
        if not self.stream:
            return {"status": "error", "message": "No data to analyze"}

        # Calculate Q1 (25th percentile) and Q3 (75th percentile) using linear interpolation
        q1 = np.percentile(self.stream, 25, method='linear')
        q3 = np.percentile(self.stream, 75, method='linear')

        # Calculate IQR and lower bound
        iqr = q3 - q1
        lower_bound = q1 - self.alpha * iqr

        # Find lower outliers
        lower_outliers = [x for x in self.stream if x < lower_bound]

        return {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "lower_outliers": lower_outliers
        }

    def print_results(self):
        """Print the analysis results."""
        print("\nIQR Analysis Results:")
        print(f"Q1 (First Quartile): {self.results['Q1']:.2f}")
        print(f"Q3 (Third Quartile): {self.results['Q3']:.2f}")
        print(f"IQR: {self.results['IQR']:.2f}")
        print(f"Lower Bound: {self.results['lower_bound']:.2f}")
        print(f"Lower Outliers: {self.results['lower_outliers']}")

# Example usage
if __name__ == "__main__":
    stream = [-1.2, -1.2, 16.3, 4.4, 16.3, -1.2, -1.2, 6.8, 2.6]
    detector = IQROutlierDetector(stream, alpha=1.9)
