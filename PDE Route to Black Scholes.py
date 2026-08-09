
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
    
    return call_price

call_price = black_scholes_call(3, 1.5, 5, 0.02, 0.3)
print(f"Call Option Price: {call_price:.4f}")

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
    
    return put_price

put_price = black_scholes_put(3, 1.5, 5, 0.02, 0.3)
print(f"Put Option Price: {put_price:.4f}")