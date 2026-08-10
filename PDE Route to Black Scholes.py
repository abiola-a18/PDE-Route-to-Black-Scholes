
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price of a European call option.

    A European call option gives the holder the right, but not the obligation, to buy an underlying asset at a 
    specified strike price (K) on or before a specified expiration date (T). The Black-Scholes model provides
    a theoretical estimate of the price of such options based on several parameters.
    
    Parameters:
    S : float
        Current stock price
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying stock (annualized)

    Returns:
    float
        Price of the European call option
    """

    import numpy as np
    from scipy.stats import norm
    # Calculate d1 and d2 using the Black-Scholes (BSM) formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculating the call option price using BSM
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return print(f"Call Option Price: {call_price:.4f}")




def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price of a European put option.

    A European put option gives the holder the right, but not the obligation, to sell an underlying asset at a 
    specified strike price (K) on or before a specified expiration date (T). The Black-Scholes model provides
    a theoretical estimate of the price of such options based on several parameters.
    
    Parameters:
    S : float
        Current stock price
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying stock (annualized)

    Returns:
    float
        Price of the European put option
    """

    import numpy as np
    from scipy.stats import norm
    # Calculate d1 and d2 using the Black-Scholes (BSM) formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculating the put option price using BSM
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return print(f"Put Option Price: {put_price:.4f}")


def Bsm_finite_differences_Call(S, K, T, r, sigma, M, N):
    """
    Calculate the Black-Scholes price of a European option using finite difference methods.

    This function uses a finite difference approach to solve the Black-Scholes partial differential equation (PDE)
    for European options. It discretizes the time and stock price dimensions to approximate the option price.

    Parameters:
    S : float
        Current stock price
    K : float
        Strike price of the option
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (annualized)
    sigma : float
        Volatility of the underlying stock (annualized)
    M : int
        Number of time steps
    N : int
        Number of stock price steps

    Returns:
    tuple
        Price of the European call and put options as a tuple (call_price, put_price)
    """

    import numpy as np

    # Discretize time and stock price dimensions
    dt = T / M  # Time step size
    dS = S / N  # Stock price step size

    # Initialize asset prices and option values at maturity
    SP = np.linspace(0, S, N + 1)  # Stock prices from 0 to S
    t = np.linspace(0, T, M + 1)  # Time steps from 0 to T

    V = np.zeros((M + 1, N + 1))  # Initialize option value matrix
    V[:, -1] = np.maximum(SP- K, 0)  # Call option payoff at maturity

    
    for j in range(M-1, -1 , -1):
        for i in range(1, N):
            a = 0.5 * dt * (sigma ** 2 * i ** 2 + r * i)
            b = ((sigma ** 2 * i ** 2 + r ) * dt +1)
            c = - 0.5 * dt * ( - sigma ** 2 * i ** 2 + r * i)
            V[i, j] = V[i+i, j+1] - a/b * V[i-1, j+1] - c/b * V[i+1, j+1]


    return print(f"Call Numeric Option Price: {V[-1, -1]:.4f}")  # Return the option price at the initial time and stock price

Bsm_finite_differences_Call(100, 100, 1, 0.05, 0.2, 100, 100)