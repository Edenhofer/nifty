from time import time

import matplotlib.pyplot as plt
import numpy as np

import nifty6 as ift

N0s, a0s, b0s, c0s = [], [], [], []

for ii in range(10, 26):
    nu = 1024
    nv = 1024
    N = int(2**ii)
    print('N = {}'.format(N))

    rng = ift.random.current_rng()
    uv = rng.uniform(-.5, .5, (N,2))
    vis = rng.normal(0., 1., N) + 1j*rng.normal(0., 1., N)

    uvspace = ift.RGSpace((nu, nv))

    visspace = ift.UnstructuredDomain(N)

    img = rng.standard_normal((nu, nv))
    img = ift.makeField(uvspace, img)

    t0 = time()
    GM = ift.GridderMaker(uvspace, eps=1e-7, uv=uv)
    vis = ift.makeField(visspace, vis)
    op = GM.getFull().adjoint
    t1 = time()
    op(img).val
    t2 = time()
    op.adjoint(vis).val
    t3 = time()
    print(t2-t1, t3-t2)
    N0s.append(N)
    a0s.append(t1 - t0)
    b0s.append(t2 - t1)
    c0s.append(t3 - t2)

print('Measure rest operator')
sc = ift.StatCalculator()
op = GM.getRest().adjoint
for _ in range(10):
    t0 = time()
    res = op(img)
    sc.add(time() - t0)
t_fft = sc.mean
print('FFT shape', res.shape)

plt.scatter(N0s, a0s, label='Gridder mr')
plt.legend()
# no idea why this is necessary, but if it is omitted, the range is wrong
plt.ylim(min(a0s), max(a0s))
plt.ylabel('time [s]')
plt.title('Initialization')
plt.loglog()
plt.savefig('bench0.png')
plt.close()

plt.scatter(N0s, b0s, color='k', marker='^', label='Gridder mr times')
plt.scatter(N0s, c0s, color='k', label='Gridder mr adjoint times')
plt.axhline(sc.mean, label='FFT')
plt.axhline(sc.mean + np.sqrt(sc.var))
plt.axhline(sc.mean - np.sqrt(sc.var))
plt.legend()
plt.ylabel('time [s]')
plt.title('Apply')
plt.loglog()
plt.savefig('bench1.png')
plt.close()
