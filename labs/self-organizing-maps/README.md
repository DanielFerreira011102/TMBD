# Questions for Self-Organizing Maps (SOMs)

- **Q1**: What is the numeric criteria that you may use to determine if a change in the algorithm produces improvements?

The numeric criteria that you may use to determine if a change in the algorithm produces improvements is the quantization error. The quantization error is the average distance between each input vector and its best matching unit. The quantization error is calculated as follows:

$$
\text{QE} = \frac{1}{N} \sum_{i=1}^{N} \text{dist}(\text{input}_i, \text{bmu}_i)
$$

where $N$ is the number of input vectors, $\text{dist}(\text{input}_i, \text{bmu}_i)$ is the distance between the input vector $\text{input}_i$ and its best matching unit $\text{bmu}_i$.

The quantization error is a measure of how well the SOM preserves the topology of the input space. The lower the quantization error, the better the SOM is at preserving the topology of the input space. Therefore, if a change in the algorithm produces improvements, the quantization error should decrease.

The quantization error can also be used to determine the optimal number of epochs for training the SOM. By monitoring the quantization error during training, you can determine when the SOM has converged and stop training.

- **Q2**: Write the version SOM1A, where you change the curve of the learning factor. Did you achieve improvements?

- **Q3**: Write the version SOM1B, where you change the curve of the deviation. Did you achieve improvements?

- **Q4**: Write the version SOM1C, where you change the change the normal distribution to other distribution of your choice. Did you achieve improvements?

- **Q5\***: Determine the mathematical conditions that ensure the convergence of equation (3) in page 14 of this slides.

$$
hc_k = \exp \left( -\frac{||x - w_k||^2}{2\sigma^2} \right)
$$

The de

- **Q6**: As explained in class, SOM can be seen as a Euler integration method for the corresponding ODE. Estimate the absolute error after N epochs.

- **Q7\***: How could you change the SOM method to use Runge-Kutta second order method? Is the improvements?

- **Q8\***: Estimate the absolute error after N epochs by using Q7.

- **Q9**: How would you combine the answers to Q1-Q8, in order to suggest an improved version?
