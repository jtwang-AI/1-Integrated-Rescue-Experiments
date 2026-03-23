import numpy as np
import matplotlib.pyplot as plt


def ltr1_ellipse1(num_steps, x_init, T, k1, k2, a, b, e0, e1, e2, e3, e4):
    eps = 1e-12

    # Initialize arrays
    px0 = np.zeros(num_steps)
    px1 = np.zeros(num_steps)
    px2 = np.zeros(num_steps)
    px3 = np.zeros(num_steps)
    px4 = np.zeros(num_steps)

    py0 = np.zeros(num_steps)
    py1 = np.zeros(num_steps)
    py2 = np.zeros(num_steps)
    py3 = np.zeros(num_steps)
    py4 = np.zeros(num_steps)

    ux0 = np.zeros(num_steps)
    ux1 = np.zeros(num_steps)
    ux2 = np.zeros(num_steps)
    ux3 = np.zeros(num_steps)
    ux4 = np.zeros(num_steps)

    uy0 = np.zeros(num_steps)
    uy1 = np.zeros(num_steps)
    uy2 = np.zeros(num_steps)
    uy3 = np.zeros(num_steps)
    uy4 = np.zeros(num_steps)

    lambda0 = np.zeros(num_steps)
    lambda1 = np.zeros(num_steps)
    lambda2 = np.zeros(num_steps)
    lambda3 = np.zeros(num_steps)
    lambda4 = np.zeros(num_steps)

    bar_gs0 = np.zeros(num_steps)
    bar_gs1 = np.zeros(num_steps)
    bar_gs2 = np.zeros(num_steps)
    bar_gs3 = np.zeros(num_steps)
    bar_gs4 = np.zeros(num_steps)

    gs0 = np.zeros(num_steps)
    gs1 = np.zeros(num_steps)
    gs2 = np.zeros(num_steps)
    gs3 = np.zeros(num_steps)
    gs4 = np.zeros(num_steps)

    # Initial conditions from x_init
    px0[0] = x_init[0]
    py0[0] = x_init[1]
    gs0[0] = x_init[2]
    lambda0[0] = x_init[3]

    px1[0] = x_init[4]
    py1[0] = x_init[5]
    gs1[0] = x_init[6]
    lambda1[0] = x_init[7]

    px2[0] = x_init[8]
    py2[0] = x_init[9]
    gs2[0] = x_init[10]
    lambda2[0] = x_init[11]

    px3[0] = x_init[12]
    py3[0] = x_init[13]
    gs3[0] = x_init[14]
    lambda3[0] = x_init[15]

    px4[0] = x_init[16]
    py4[0] = x_init[17]
    gs4[0] = x_init[18]
    lambda4[0] = x_init[19]

    J = np.array([[0.0, 1.0], [-1.0, 0.0]])

    for t in range(num_steps - 1):
        # Formation errors
        bar_gs0[t] = 0.0
        bar_gs1[t] = gs1[t] - gs0[t]
        bar_gs2[t] = gs2[t] - gs1[t]
        bar_gs3[t] = gs3[t] - gs2[t] + gs3[t] - gs4[t]
        bar_gs4[t] = gs4[t] - gs1[t]

        # Formation movement errors
        g_xi0 = 0.04
        g_xi1 = -k2 * T * bar_gs1[t] + g_xi0
        g_xi2 = -k2 * T * bar_gs2[t] + g_xi1
        g_xi4 = -k2 * T * bar_gs4[t] + g_xi1
        g_xi3 = -k2 * T * bar_gs3[t] + (g_xi2 + g_xi4) / 2.0

        # Gradients
        nabla_lambda_0 = -np.array([px0[t] / ((e0 * a) ** 2), py0[t] / ((e0 * b) ** 2)])
        nabla_lambda_1 = -np.array([px1[t] / ((e1 * a) ** 2), py1[t] / ((e1 * b) ** 2)])
        nabla_lambda_2 = -np.array([px2[t] / ((e2 * a) ** 2), py2[t] / ((e2 * b) ** 2)])
        nabla_lambda_3 = -np.array([px3[t] / ((e3 * a) ** 2), py3[t] / ((e3 * b) ** 2)])
        nabla_lambda_4 = -np.array([px4[t] / ((e4 * a) ** 2), py4[t] / ((e4 * b) ** 2)])

        norm_nabla_lambda_0 = max(np.linalg.norm(nabla_lambda_0), eps)
        norm_nabla_lambda_1 = max(np.linalg.norm(nabla_lambda_1), eps)
        norm_nabla_lambda_2 = max(np.linalg.norm(nabla_lambda_2), eps)
        norm_nabla_lambda_3 = max(np.linalg.norm(nabla_lambda_3), eps)
        norm_nabla_lambda_4 = max(np.linalg.norm(nabla_lambda_4), eps)

        # Safe square roots
        s0 = max(px0[t] ** 2 + py0[t] ** 2 + k1 * T * (e0 * a) ** 2 * lambda0[t], 0.0)
        s1 = max(px1[t] ** 2 + py1[t] ** 2 + k1 * T * (e1 * a) ** 2 * lambda1[t], 0.0)
        s2 = max(px2[t] ** 2 + py2[t] ** 2 + k1 * T * (e2 * a) ** 2 * lambda2[t], 0.0)
        s3 = max(px3[t] ** 2 + py3[t] ** 2 + k1 * T * (e3 * a) ** 2 * lambda3[t], 0.0)
        s4 = max(px4[t] ** 2 + py4[t] ** 2 + k1 * T * (e4 * a) ** 2 * lambda4[t], 0.0)

        r0 = np.sqrt(px0[t] ** 2 + py0[t] ** 2)
        r1 = np.sqrt(px1[t] ** 2 + py1[t] ** 2)
        r2 = np.sqrt(px2[t] ** 2 + py2[t] ** 2)
        r3 = np.sqrt(px3[t] ** 2 + py3[t] ** 2)
        r4 = np.sqrt(px4[t] ** 2 + py4[t] ** 2)

        root0 = np.sqrt(s0)
        root1 = np.sqrt(s1)
        root2 = np.sqrt(s2)
        root3 = np.sqrt(s3)
        root4 = np.sqrt(s4)

        # Tangential / normal speeds
        vT0 = np.sin(g_xi0) * root0 / T
        vT1 = np.sin(g_xi1) * root1 / T
        vT2 = np.sin(g_xi2) * root2 / T
        vT3 = np.sin(g_xi3) * root3 / T
        vT4 = np.sin(g_xi4) * root4 / T

        vN0 = (r0 - np.cos(g_xi0) * root0) / T
        vN1 = (r1 - np.cos(g_xi1) * root1) / T
        vN2 = (r2 - np.cos(g_xi2) * root2) / T
        vN3 = (r3 - np.cos(g_xi3) * root3) / T
        vN4 = (r4 - np.cos(g_xi4) * root4) / T

        # Normal vectors
        N0 = nabla_lambda_0 / norm_nabla_lambda_0
        N1 = nabla_lambda_1 / norm_nabla_lambda_1
        N2 = nabla_lambda_2 / norm_nabla_lambda_2
        N3 = nabla_lambda_3 / norm_nabla_lambda_3
        N4 = nabla_lambda_4 / norm_nabla_lambda_4

        # Tangent vectors
        T0v = J @ N0
        T1v = J @ N1
        T2v = J @ N2
        T3v = J @ N3
        T4v = J @ N4

        # Rotation matrices
        R0 = np.linalg.inv(np.vstack([N0, T0v]))
        R1 = np.linalg.inv(np.vstack([N1, T1v]))
        R2 = np.linalg.inv(np.vstack([N2, T2v]))
        R3 = np.linalg.inv(np.vstack([N3, T3v]))
        R4 = np.linalg.inv(np.vstack([N4, T4v]))

        # Control inputs
        u0 = R0 @ np.array([vN0, vT0])
        u1 = R1 @ np.array([vN1, vT1])
        u2 = R2 @ np.array([vN2, vT2])
        u3 = R3 @ np.array([vN3, vT3])
        u4 = R4 @ np.array([vN4, vT4])

        ux0[t], uy0[t] = u0[0], u0[1]
        ux1[t], uy1[t] = u1[0], u1[1]
        ux2[t], uy2[t] = u2[0], u2[1]
        ux3[t], uy3[t] = u3[0], u3[1]
        ux4[t], uy4[t] = u4[0], u4[1]

        # Update tracking errors
        lambda0[t + 1] = (1 - k1 * T) * lambda0[t]
        lambda1[t + 1] = (1 - k1 * T) * lambda1[t]
        lambda2[t + 1] = (1 - k1 * T) * lambda2[t]
        lambda3[t + 1] = (1 - k1 * T) * lambda3[t]
        lambda4[t + 1] = (1 - k1 * T) * lambda4[t]

        # Update positions
        px0[t + 1] = px0[t] + T * ux0[t]
        py0[t + 1] = py0[t] + T * uy0[t]

        px1[t + 1] = px1[t] + T * ux1[t]
        py1[t + 1] = py1[t] + T * uy1[t]

        px2[t + 1] = px2[t] + T * ux2[t]
        py2[t + 1] = py2[t] + T * uy2[t]

        px3[t + 1] = px3[t] + T * ux3[t]
        py3[t + 1] = py3[t] + T * uy3[t]

        px4[t + 1] = px4[t] + T * ux4[t]
        py4[t + 1] = py4[t] + T * uy4[t]

        # Update generalized arc lengths
        gs0[t + 1] = gs0[t] + g_xi0
        gs1[t + 1] = gs1[t] + g_xi1
        gs2[t + 1] = gs2[t] + g_xi2
        gs3[t + 1] = gs3[t] + g_xi3
        gs4[t + 1] = gs4[t] + g_xi4

    # Keep the same column order as your original plotting code expects
    x_hist = np.column_stack([
        px1, py1,          # 0,1
        px2, py2,          # 2,3
        px3, py3,          # 4,5
        px4, py4,          # 6,7
        lambda1, lambda2, lambda3, lambda4,  # 8,9,10,11
        gs0, gs1, gs2, gs3, gs4,             # 12,13,14,15,16
        px0, py0           # 17,18
    ])

    return x_hist


def build_demo_initial_state(a=3, b=3, e0=0.5, e1=1.0, e2=1.5, e3=2.0, e4=2.5):
    p_0x_0 = e0 * a * np.cos(np.pi / 2)
    p_0y_0 = e0 * b * np.sin(np.pi / 2)
    lambda_0_0 = 1 - (p_0x_0 / (a * e0)) ** 2 - (p_0y_0 / (b * e0)) ** 2
    gs_0_0 = np.arctan2(p_0y_0, p_0x_0)

    p_1x_0 = 1.15 * e1 * a * np.cos(np.pi / 4)
    p_1y_0 = 1.15 * e1 * b * np.sin(np.pi / 4)
    lambda_1_0 = 1 - (p_1x_0 / (a * e1)) ** 2 - (p_1y_0 / (b * e1)) ** 2
    gs_1_0 = np.arctan2(p_1y_0, p_1x_0)

    p_2x_0 = 1.12 * e2 * a * np.cos(np.pi / 5)
    p_2y_0 = 1.12 * e2 * b * np.sin(np.pi / 5)
    lambda_2_0 = 1 - (p_2x_0 / (a * e2)) ** 2 - (p_2y_0 / (b * e2)) ** 2
    gs_2_0 = np.arctan2(p_2y_0, p_2x_0)

    p_3x_0 = 1.1 * e3 * a * np.cos(np.pi / 6)
    p_3y_0 = 1.1 * e3 * b * np.sin(np.pi / 6)
    lambda_3_0 = 1 - (p_3x_0 / (a * e3)) ** 2 - (p_3y_0 / (b * e3)) ** 2
    gs_3_0 = np.arctan2(p_3y_0, p_3x_0)

    p_4x_0 = 1.2 * e4 * a * np.cos(np.pi / 7)
    p_4y_0 = 1.2 * e4 * b * np.sin(np.pi / 7)
    lambda_4_0 = 1 - (p_4x_0 / (a * e4)) ** 2 - (p_4y_0 / (b * e4)) ** 2
    gs_4_0 = np.arctan2(p_4y_0, p_4x_0)

    return np.array([
        p_0x_0, p_0y_0, gs_0_0, lambda_0_0,
        p_1x_0, p_1y_0, gs_1_0, lambda_1_0,
        p_2x_0, p_2y_0, gs_2_0, lambda_2_0,
        p_3x_0, p_3y_0, gs_3_0, lambda_3_0,
        p_4x_0, p_4y_0, gs_4_0, lambda_4_0
    ])


def run_demo(show=True):
    a = 3
    b = 3
    e0 = 0.5
    e1 = 1.0
    e2 = 1.5
    e3 = 2.0
    e4 = 2.5
    T = 0.01
    k1 = 30
    k2 = 15
    t_final = 51

    x_init = build_demo_initial_state(a, b, e0, e1, e2, e3, e4)
    x = ltr1_ellipse1(t_final, x_init, T, k1, k2, a, b, e0, e1, e2, e3, e4)

    plt.figure()
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 8
    plt.plot(x[:, 17], x[:, 18], ':k', linewidth=2, label='p_0')
    plt.plot(x[:, 0], x[:, 1], ':c', linewidth=2, label='p_1')
    plt.plot(x[:, 2], x[:, 3], ':b', linewidth=2, label='p_2')
    plt.plot(x[:, 4], x[:, 5], ':g', linewidth=2, label='p_3')
    plt.plot(x[:, 6], x[:, 7], ':r', linewidth=2, label='p_4')
    plt.legend()

    mark_idx = [0, 10, 30, 50]
    for idx in mark_idx:
        if idx < len(x):
            plt.plot(x[idx, 0], x[idx, 1], '*c' if idx == 0 else 'oc')
            plt.plot(x[idx, 2], x[idx, 3], '*b' if idx == 0 else 'ob')
            plt.plot(x[idx, 4], x[idx, 5], '*g' if idx == 0 else 'og')
            plt.plot(x[idx, 6], x[idx, 7], '*r' if idx == 0 else 'or')
            plt.plot(x[idx, 17], x[idx, 18], '*k' if idx == 0 else 'ok')

    theta = np.linspace(0, 2 * np.pi, 200)
    for scale in [e0, e1, e2, e3, e4]:
        plt.plot(a * scale * np.cos(theta), b * scale * np.sin(theta), 'k' if scale != e0 else '--k')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectories')

    n_lambda = min(21, x.shape[0])
    t1 = np.arange(n_lambda)
    plt.figure()
    plt.plot(t1, x[:n_lambda, 8], '-oc', label=r'$\lambda_1$')
    plt.plot(t1, x[:n_lambda, 9], '-ob', label=r'$\lambda_2$')
    plt.plot(t1, x[:n_lambda, 10], '-og', label=r'$\lambda_3$')
    plt.plot(t1, x[:n_lambda, 11], '-or', label=r'$\lambda_4$')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel(r'$\lambda_i$')
    plt.title('Tracking errors')

    n_gs = min(31, x.shape[0])
    t2 = np.arange(n_gs)
    plt.figure()
    plt.plot(t2, x[:n_gs, 13] - x[:n_gs, 12], '-oc', label=r'$\xi_1 - \xi_0$')
    plt.plot(t2, x[:n_gs, 14] - x[:n_gs, 12], '-ob', label=r'$\xi_2 - \xi_0$')
    plt.plot(t2, x[:n_gs, 15] - x[:n_gs, 12], '-og', label=r'$\xi_3 - \xi_0$')
    plt.plot(t2, x[:n_gs, 16] - x[:n_gs, 12], '-or', label=r'$\xi_4 - \xi_0$')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel(r'$\xi_i - \xi_0$')
    plt.title('Generalized arc-length differences')

    if show:
        plt.show()
    return x


if __name__ == "__main__":
    run_demo(show=True)


