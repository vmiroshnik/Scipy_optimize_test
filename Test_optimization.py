import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.optimize as opt


#For reproducibility. Comment it out for randomness
np.random.seed(413)

#Uncoomment and comment next line if you want to try random init
# clean_random_weights = scipy.random.standard_normal((2, 1))
clean_random_weights = np.asarray([[-2.8], [-2.5]]).astype(dtype=np.float64)
W = clean_random_weights



levels = [x/4.0 for x in range(-8, 2*12, 1)] + [6.25, 6.5, 6.75, 7] + \
         list(range(8, 20, 1))
levels = np.asarray(levels)
#
O_simple_quad = lambda W: (W**2).sum()
O_wobbly = lambda W: (np.power(W, 2)).sum()/3 + np.abs(W[0])*np.sqrt(np.abs(W[0]) + 0.1) + 3*np.sin(W.sum()) + 3.0 + 8*np.exp(-2*((W[0] + 1)**2+(W[1] + 2)**2))
O_basins_and_walls = lambda W: (W**2).sum()/2 + np.sin(W[0]*4)**2
O_ripple = lambda W: (W**2).sum()/3 + (np.sin(W[0]*20)**2 + np.sin(W[1]*20)**2)/15
O_giant_plateu = lambda W: 4*(1-np.exp(-((W[0])**2+(W[1])**2)))
O_hills_and_canyon = lambda W: (W**2).sum()/3 + \
                     3*np.exp(-((W[0] + 1)**2+(W[1] + 2)**2)) + \
                       np.exp(-1.5*(2*(W[0] + 2)**2+(W[1] - 0.5)**2)) + \
                     3*np.exp(-1.5*((W[0] - 1)**2+2*(W[1] + 1.5)**2)) + \
                     1.5*np.exp(-((W[0] + 1.5)**2+3*(W[1] + 0.5)**2)) + \
                     4*(1 - np.exp(-((W[0] + W[1])**2)))

O_two_minimums = lambda W: 4-0.5*np.exp(-((W[0] + 2.5)**2+(W[1] + 2.5)**2))-3*np.exp(-((W[0])**2+(W[1])**2))

cross_method_testsuit = [
                (opt.minimize,        "BFGS",         {'disp': True,'maxiter': 100}),
                (opt.minimize,        "CG",           {'disp': True, 'maxiter': 100}),
                (opt.minimize,        "Powell",       {'disp': True, 'maxiter': 100}),
                (opt.minimize,        "Nelder-Mead",  {'disp': True, 'maxiter': 100}),
                # (opt.minimize,        "Newton-CG",  {'disp': True, 'maxiter': 100}), # Jacobian is required for Newton-CG method
                # (nesterov_momentum, "nesterov",     {"learning_rate": 0.01}),
                # (rmsprop,           "rmsprop",      {"learning_rate": 0.25}),
                # (adadelta,          "adadelta",     {"learning_rate": 100.0}),
                # (adagrad,           "adagrad",      {"learning_rate": 1.0}),
                # (adam,              "adam",         {"learning_rate": 0.25})

            ]
#
for O, plot_label in [
           (O_wobbly, "Wobbly"),
           (O_basins_and_walls, "Basins_and_walls"),
           (O_giant_plateu, "Giant_plateu"),
           (O_hills_and_canyon, "Hills_and_canyon"),
           (O_two_minimums, "Bad_init")
        ]:
    #
    history = {}
    for method, history_mark, kwargs_to_method in cross_method_testsuit:
        res_x = []
        if history_mark in ('CG', 'BFGS', 'Newton-CG'):
            method(O, W,
                   method=history_mark,
                   jac=False,
                   tol=0.00001,
                   callback=res_x.append,
                   options=kwargs_to_method
                   )
        else:
            method(O, W,
                   method=history_mark,
                   tol=0.00001,
                   callback=res_x.append,
                   options=kwargs_to_method)
        init_l = len(res_x)
        history_mark += '_' + str(init_l)
        history[history_mark] = [clean_random_weights]
        while len(res_x) < 125:
            tmp = res_x[-1]
            res_x.append(tmp)

        for i in range(125):
            par = res_x[i]
            result_val = O(par)
            history[history_mark].append(par)

        print("-------- DONE {}-------".format(history_mark))

    delta = 0.05
    mesh = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(mesh, mesh)

    Z = []
    for y in mesh:
        z = []
        for x in mesh:
            z.append(O(np.array([x, y])))
        Z.append(z)
    Z = np.asarray(Z)

    print("-------- BUILT MESH -------")

    fig, ax = plt.subplots(figsize=[12.0, 8.0])
    CS = ax.contour(X, Y, Z, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(plot_label)

    nphistory = []
    for key in history:
        # print(history[key][0])
        nphistory.append(
                [np.asarray([h[0] for h in history[key]]),
                 np.asarray([h[1] for h in history[key]]),
                 key]
            )

    lines = []
    for nph in nphistory:
        lines += ax.plot(nph[0], nph[1], label=nph[2])
    leg = plt.legend()

    plt.savefig('animation\\' + plot_label + '_final.png')

    def animate(i):
        for line, hist in zip(lines, nphistory):
            if len(hist[0][:i]) != len(hist[1][:i]):
                print(len(hist[0][:i]), len(hist[1][:i]), '\n')
            line.set_xdata(hist[0][:i])
            line.set_ydata(hist[1][:i])
        return lines

    for i in range(120):
        animate(i)


    def init():
        for line, hist in zip(lines, nphistory):
            line.set_ydata(np.ma.array(hist[0], mask=True))
        return lines

    ani = animation.FuncAnimation(fig, animate, 120, init_func=init,
                                  interval=100, repeat_delay=0, blit=True, repeat=True)

    print("-------- WRITING ANIMATION -------")
    # plt.show(block=True) #Uncoomment and comment next line if you just want to watch
    ani.save('animation\\' + plot_label + '.mp4', writer='ffmpeg_file', fps=5)

    print("-------- DONE {} -------".format(plot_label))
