class BaseAlg:
    # Input stream and results
    stream = []                 # List of values in the stream
    results = {"outliers": [], "processed": []}  # Dictionary to store results
    current_value = 0           # Current value being processed
    
    # Streaming statistics
    count = 0                   # Number of values processed (n)
    mean = 0.0                  # Running mean (μ)
    sum_of_square_diffs = 0.0   # Running sum of squared differences (M2)
    z_threshold = 1           # Threshold for Z-score (alpha)

    def alg(self):
        """
        Update the mean and variance incrementally using Welford's Algorithm.
        Detect and flag outliers based on the Z-score threshold.
        """
        if self.count == 0:
            # Initialize on first value
            self.count = 1
            self.mean = self.current_value
            self.sum_of_square_diffs = 0.0
            self.results["processed"].append(self.current_value)
        else:
            # Incrementally update statistics
            self.count += 1
            delta = self.current_value - self.mean  # Difference from previous mean
            self.mean += delta / self.count         # Update mean
            delta2 = self.current_value - self.mean # Difference from updated mean
            self.sum_of_square_diffs += delta * delta2  # Update sum of squared differences (M2)
            
            # Calculate standard deviation
            variance = self.sum_of_square_diffs / (self.count - 1)
            std_dev = variance ** 0.5

            # Calculate outlier bounds based on Z-score
            lower_bound = self.mean - self.z_threshold * std_dev
            upper_bound = self.mean + self.z_threshold * std_dev

            # Check if current value is an outlier
            if self.current_value < lower_bound or self.current_value > upper_bound:
                self.results["outliers"].append(self.current_value)
            else:
                self.results["processed"].append(self.current_value)

    def verify(self):
        """
        Verify the results by detecting outliers using the Z-score method on the complete dataset.
        Returns a dictionary comparing streaming and batch results.
        """
        if not self.stream:
            return {"status": "error", "message": "No data to verify"}

        # Calculate mean and standard deviation for the entire dataset
        batch_mean = sum(self.stream) / len(self.stream)
        squared_diff_sum = sum((x - batch_mean) ** 2 for x in self.stream)
        batch_std = (squared_diff_sum / (len(self.stream) - 1)) ** 0.5

        # Detect outliers using batch processing
        batch_results = {"outliers": [], "processed": []}
        for value in self.stream:
            z_score = (value - batch_mean) / batch_std if batch_std > 0 else 0
            if abs(z_score) > self.z_threshold:
                batch_results["outliers"].append(value)
            else:
                batch_results["processed"].append(value)

        # Compare streaming and batch results
        verification_results = {
            "status": "success",
            "streaming_outliers": sorted(self.results["outliers"]),
            "batch_outliers": sorted(batch_results["outliers"]),
            "streaming_processed": sorted(self.results["processed"]),
            "batch_processed": sorted(batch_results["processed"]),
            "matches": (
                sorted(self.results["outliers"]) == sorted(batch_results["outliers"]) and
                sorted(self.results["processed"]) == sorted(batch_results["processed"])
            )
        }
        
        print("Verification Results:")
        print(f"Streaming Outliers: {verification_results['streaming_outliers']}")
        print(f"Batch Outliers: {verification_results['batch_outliers']}")
        print(f"Results Match: {verification_results['matches']}")
        
        return verification_results

class StreamAlg(BaseAlg):
    def __init__(self, stream):
        self.stream = stream
        self.exec()
        self.verify()

    def exec(self):
        """ Process each value in the stream using the algorithm. """
        for value in self.stream:
            self.current_value = value
            self.alg()
        print(self.results)

# Example usage
if __name__ == "__main__":
    stream = [-1.2, -1.2, 16.3, 4.4, 16.3, -1.2, -1.2, 6.8, 2.6]
    SA = StreamAlg(stream)