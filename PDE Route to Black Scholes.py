import matplotlib.pyplot as plt
import numpy as np

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

def BSMFDC(S, K, T, r, sigma, M, N, Smax_factor = 3.0):
    """
    Calculate the Black-Scholes price of a European call option using finite difference methods.

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
    S_max = Smax_factor * max(S, K)
    dt = T / M  # Time step size
    dS = S_max / N  # Stock price step size

    # Initialize asset prices and option values at maturity
    SP = np.linspace(0, S_max, N + 1)  # Stock prices from 0 to S
    V = np.linspace(0, T, M + 1)  # Time steps from 0 to T

    V = np.zeros((N + 1, M + 1))  # Initialize option value matrix
    V[:, -1] = np.maximum(SP - K, 0)  # Call option payoff at maturity

    a = 0.5*dt*(sigma**2*SP**2/dS**2 - r*SP/dS)
    b = 1 - dt*(sigma**2*SP**2/dS**2 + r)
    c = 0.5*dt*(sigma**2*SP**2/dS**2 + r*SP/dS)

    for j in range(M-1, -1 , -1):
        for i in range(1, N):
            V[i,j] = a[i]*V[i-1,j+1] + b[i]*V[i,j+1] + c[i]*V[i+1,j+1]

    return V[:,0], SP

def BlackScholesCallComparison(S, K, T, r, sigma, M, N, Smax_factor = 3.0):
    """
    Compare the Black-Scholes price of a European call option using finite difference methods with the analytical solution.

    This function computes the option price using both the finite difference method and the analytical Black-Scholes formula,
    and then compares the two results.

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
        Number of time steps for finite difference method
    N : int
        Number of stock price steps for finite difference method

    Returns:
    Graph
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Compute option values across the stock price grid using finite difference method
    V_t0, SP = BSMFDC(S, K, T, r, sigma, M, N, Smax_factor)

    # Compute analytical Black-Scholes call option price
    call_price_analytical = black_scholes_call(S, K, T, r, sigma)

    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(SP, V_t0, label='Option Value at t=0 (Finite Difference)', color='blue', linewidth=2)
    plt.plot(SP, np.maximum(SP - K, 0), label='Payoff at Maturity (t=T)', color='black', linestyle='--')
    plt.axvline(x=S, color='red', linestyle=':', label=f'Current Spot Price (S={S})')

    plt.title('European Call Option Price vs. Stock Price', fontsize=12, fontweight='bold')
    plt.xlabel('Stock Price (S)', fontsize=10)
    plt.ylabel('Option Price V(S, t=0)', fontsize=10)
    plt.xlim(50, 150)
    plt.ylim(-2, 55)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    return fig, ax

def BSMFDC3D(S, K, T, r, sigma, M, N, Smax_factor = 3.0):
    """
    Calculate the Black-Scholes price of a European call option using finite difference methods.

    !!!3D Adjusted!!!
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
    S_max = Smax_factor * max(S, K)
    dt = T / M  # Time step size
    dS = S_max / N  # Stock price step size

    # Initialize asset prices and option values at maturity
    SP = np.linspace(0, S_max, N + 1)  # Stock prices from 0 to S
    V = np.linspace(0, T, M + 1)  # Time steps from 0 to T

    V = np.zeros((N + 1, M + 1))  # Initialize option value matrix
    V[:, -1] = np.maximum(SP - K, 0)  # Call option payoff at maturity

    a = 0.5*dt*(sigma**2*SP**2/dS**2 - r*SP/dS)
    b = 1 - dt*(sigma**2*SP**2/dS**2 + r)
    c = 0.5*dt*(sigma**2*SP**2/dS**2 + r*SP/dS)

    for j in range(M-1, -1 , -1):
        for i in range(1, N):
            V[i,j] = a[i]*V[i-1,j+1] + b[i]*V[i,j+1] + c[i]*V[i+1,j+1]

    return V, SP

def BlackScholesPutComparison3D(S, K, T, r, sigma, M, N, Smax_factor = 3.0):
    
    """
    Plot the 3D surface of the European call option price as a function of stock price and time to expiration using finite difference methods.
    """
    
    # 2. Create 2D Meshgrid for Plotting
    # T_mesh shape: (N+1, M+1), S_mesh shape: (N+1, M+1)
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    V_t0, SP = BSMFDC3D(S, K, T, r, sigma, M, N, Smax_factor)

    t = np.linspace(0, T, M + 1)  # Time steps from 0 to T
    T_mesh, S_mesh = np.meshgrid(t, SP)

    # 3. 3D Plotting
    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(S_mesh, T_mesh, V_t0, cmap='viridis', edgecolor='none', alpha=0.9)

    # Labels and View Angle
    ax.set_title('3D European Call Option Surface V(S, t)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Stock Price (S)', fontsize=10, labelpad=10)
    ax.set_ylabel('Time t (Years)', fontsize=10, labelpad=10)
    ax.set_zlabel('Option Value V', fontsize=10, labelpad=10)

    # Zoom in on relevant stock price region
    ax.set_xlim(50, 150)
    ax.set_zlim(0, 60)

    # Rotate perspective (elev=elevation angle, azim=azimuthal angle)
    ax.view_init(elev=25, azim=-120)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Option Value')
    plt.tight_layout()
    plt.show()

    return fig, ax

BlackScholesPutComparison3D(S=100, K=100, T=1.0, r=0.05, sigma=0.2, M=500, N=100)