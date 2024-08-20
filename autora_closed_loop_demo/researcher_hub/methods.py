from scipy.stats import norm


def d_prime(hits, false_alarms):
    """
    Calculates the d-prime based on the hits.

    input:
    hits (int): number of hits
    false_alarms (int): number of false alarms

    returns:
    calculated d-prime value

    Example:
        >>> #hits = 1
        >>> #false_alarms = 1
        >>> #d_prime(hits, false_alarms)
        #np.float64(0.0)
    """

    hit_rate = hits / (hits + false_alarms)
    false_alarm_rate = 1 - hit_rate

    # Adjust hit rate and false alarm rate for extreme values, i.e. 1 or 0
    if hit_rate == 1:
        hit_rate = 1 - 1e-10
    elif hit_rate == 0:
        hit_rate = 1e-10

    if false_alarm_rate == 1:
        false_alarm_rate = 1 - 1e-10
    elif false_alarm_rate == 0:
        false_alarm_rate = 1e-10

    # Calculate d-prime based on z-scores of hit and false alarm rate
    d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)

    return d_prime
