
## Note on price_model.py

- GBM model has the equation:

$\log(x_T) \sim \mathcal{N}(\log(x_S)+(\mu-\frac{\sigma^2}{2})(T-S), \sigma^2(T-S))$

- Then, we can use the repramatrization trick:
- $S=\log(x_T)$
- Then, we can write
- $S=\mu + \sigma Z$

Having $Z$ as normal distributaion with mean 0 adn sd eqals to 1





