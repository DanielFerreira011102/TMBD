# Calculate the derivative for w_k(t) + \alpha * hc_k(t) * (x(t) - w_k(t))
# hc_k(t) = \exp \left( - \frac{||x(t) - w_k(t)||^2}{2 \sigma^2} \right)

# => max|g'(x)| <= 1
# => max|1 - \alpha * hc_k(t)| <= 1
# => 0 <= \alpha * hc_k(t) <= 2
# \alpha(0) <= 2 / hc_k(t)