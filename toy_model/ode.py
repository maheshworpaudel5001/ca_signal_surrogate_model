def ode(t, N, k):
    return -k * N**2


def analytical_solution(t, N0, k):
    return N0 / (1 + (k * N0 * t))
