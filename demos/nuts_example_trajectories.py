# %%
#
# WARNING: This code does not behave deterministically when compiled.
# this is probably due to an issue with host_callback
# concretely it just stops adding points to the debug list after some random number of leapfrog steps
# (currently not able to reproduce the issue but it did occur previously)
#

from functools import partial
import jax.numpy as np
import matplotlib
import matplotlib.pyplot as plt
import jifty1 as jft
from jifty1 import hmc

# %%
hmc._DEBUG_FLAG = True

# %%
cov = np.array([10., 1.])

potential_energy = lambda q: np.sum(0.5 * q**2 / cov)

initial_position = np.array([1., 1.])

sampler = hmc.NUTSChain(
        initial_position = initial_position, 
        potential_energy = potential_energy,
        diag_mass_matrix = 1.,
        eps = 0.12,
        maxdepth = 15,
        rngseed = 48,
        compile = True,
        dbg_info = True
)

# %%
hmc._DEBUG_STORE = []
hmc._DEBUG_TREE_END_IDXS = []
hmc._DEBUG_SUBTREE_END_IDXS = []

(_last_pos, _key, pos, unintegrated_momenta, momentum_samples,
 depths, trees) = (sampler.generate_n_samples(5))

#plt.hist(depths)

# %%
debug_pos = np.array([qp.position for qp in hmc._DEBUG_STORE])
print(len(debug_pos))

# %%
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax = plt.gca()
ellipse = matplotlib.patches.Ellipse(xy=(0, 0), width=np.sqrt(cov[0]), height=np.sqrt(cov[1]), 
                        edgecolor='k', fc='None', lw=1)
ax.add_patch(ellipse)

color_idx = 0
start_and_end_idxs = zip([0,] + hmc._DEBUG_SUBTREE_END_IDXS[:-1], hmc._DEBUG_SUBTREE_END_IDXS)
for start_idx, end_idx in start_and_end_idxs:
    slice = debug_pos[start_idx:end_idx]
    ax.plot(slice[:,0], slice[:,1], '-o', markersize=1, linewidth=0.5, color=colors[color_idx % len(colors)])
    if end_idx in hmc._DEBUG_TREE_END_IDXS:
        color_idx = (color_idx + 1) % len(colors)

ax.scatter(pos[:,0], pos[:,1], marker='x', color='k', label='samples')
ax.scatter(initial_position[0], initial_position[1], label='starting position')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

fig_width_pt = 426 # pt (a4paper, and such)
# fig_width_pt = 360 # pt
inches_per_pt = 1 / 72.27
fig_width_in = 0.9 * fig_width_pt * inches_per_pt
fig_height_in = fig_width_in * 0.618
fig_dims = (fig_width_in, fig_height_in)

plt.tight_layout()
plt.show()
plt.savefig("trajectories.pdf", bbox_inches='tight')
# %%
