import torch
import perceval as pcvl
import perceval.components as comp
from torch.nn import AdaptiveAvgPool2d

def tensor_decomposition(C: torch.Tensor, N: int):
    """
    Given a flattened vector C_1d of length N^2,
    find A, B in R^N (no gradient descent) such that
    A ⊗ B approximates C in the rank-1 (best in Frobenius norm) sense.

    Args:
        C (torch.Tensor): shape (N^2,1). The target vector data.
        N (int): sqrt of length of C.

    Returns:
        A, B (torch.Tensor, torch.Tensor): A ⊗ B = C
        Each shape approx_2d (torch.Tensor): The NxN rank-1 approximation of C.
    """
    # 1) Reshape to NxN
    C_2d = C.view(N, N)

    # 2) SVD
    # C_2d = U @ diag(S) @ V^T
    # torch.linalg.svd returns (U, S, Vh) with Vh = V^T
    U, S, Vh = torch.linalg.svd(C_2d)

    # 3) Extract largest singular value and vectors
    sigma = S[0]  # scalar
    u = U[:, 0]  # left singular vector
    v = Vh[0, :]  # right singular vector is V^T row

    # 4) Construct rank-1 factors:
    #    The best rank-1 approximation can be written as: sigma * u * v^T
    #    But to show usage of torch.kron, we can define:
    #        A = sqrt(sigma)*u, B = sqrt(sigma)*v
    #    so that A ⊗ B = (sqrt(sigma)*u) ⊗ (sqrt(sigma)*v)
    #                   = sigma * (u ⊗ v).
    sqrt_sigma = torch.sqrt(sigma)
    A = sqrt_sigma * u
    B = sqrt_sigma * v
    return A, B


def feature_map_distribution(u, samples, postselect):
    C1 = feature_map_circuit(u)
    proc = pcvl.Processor("SLOS", u.size(0))
    proc.set_circuit(C1)
    proc.min_detected_photons_filter(postselect)
    proc.thresholded_output(True)
    state_list = [1] + [0]*(u.size(0)-1)
    proc.with_input(pcvl.BasicState(state_list))
    sampler = pcvl.algorithm.Sampler(proc, max_shots_per_call=samples)
    res = sampler.probs(samples)
    return res


def feature_map_quantum_state(distribution, reduced_size):
    """
    convert BSD probabiltiy distribution to vector state
    :param distribution: "results" key of the class BSDistribution
    :param reduced_size:
    :return:
    """
    keys = list(distribution.keys())
    probabilities = list(distribution.values())
    indices = []
    for state in keys:
        state_vector = list(state)
        index = state_vector.index(1)
        indices.append(index)

    vector = [0] * reduced_size
    for idx, prob in zip(indices, probabilities):
        vector[idx] = prob
    return torch.tensor(vector, dtype=torch.float64)


def feature_map_unitary(u):
    """
    Feature map unitary matrix convert classic information to quantum,
    create a unitary matrix U such that U|psi_0>=|v_1> where u = |v_1>
    i.e., the first column of U is u using QR decomposition.
    :param u: Input vector (normalized quantum state).
    :return: Unitary matrix U.
    """
    u = u / torch.norm(u)
    n = len(u)
    random_matrix = torch.eye(n, dtype=torch.float64)  # Use higher precision
    random_matrix[:, 0] = u
    Q, R = torch.linalg.qr(random_matrix)
    if torch.dot(Q[:, 0], u) < 0:
        Q[:, 0] = -Q[:, 0]
    identity_check = torch.allclose(Q.T @ Q, torch.eye(n, dtype=torch.float64), atol=1e-12)
    if not identity_check:
        raise ValueError("Constructed matrix is not unitary.")
    return Q


def feature_map_circuit(u):
    """
    get the perceval circuit that does the feature map
    :param u: |v_i>, one dimensional decomposition of classic tensor
    :return: the circuit that does the feature map
    """
    U = feature_map_unitary(u).numpy()
    ub = comp.BS(theta=pcvl.P("theta")) // comp.PS(phi=pcvl.P("phi"))
    return pcvl.Circuit.decomposition(U, ub, shape=pcvl.InterferometerShape.TRIANGLE)


def simulate_quantum_image(classic_image, reduced_size, samples, postselect, device):
    """
    This method simualte the quantum state we can use
    :param classic_image: a MNIST 28*28 image
    :param reduced_size: we can reduce the image size to optimise the computational performance
    :return: a quantum image we can use (need to flatten)
    """
    images = classic_image.squeeze()[2:-2, 2:-2].unsqueeze(0)
    adaptive_avg_pool = AdaptiveAvgPool2d((reduced_size, reduced_size))
    images = adaptive_avg_pool(images).to(device).squeeze() # here we obtain the classic image we really used
    # images = (images > 0.4).float() # optinal
    A, B = tensor_decomposition(images.clone().detach(), reduced_size)
    A, B = A.to(torch.float64), B.to(torch.float64)
    res_A = feature_map_distribution(A, samples, postselect)
    res_B = feature_map_distribution(B, samples, postselect)
    distribution_A = res_A["results"]
    distribution_B = res_B["results"]
    vect_A = feature_map_quantum_state(distribution_A, reduced_size)
    vect_B = feature_map_quantum_state(distribution_B, reduced_size)
    quantum_image = torch.kron(vect_A, vect_B).reshape(reduced_size,reduced_size)
    return quantum_image
