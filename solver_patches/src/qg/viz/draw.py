import numpy as np
import matplotlib.pyplot as plt


def vorticity_plots(field,timestep, time_params):
    fig, ax = plt.subplots()  # Create a figure and axis

    cax = ax.imshow(field[:, :, timestep,0], cmap='seismic', origin='lower', vmax=10, vmin=-10, 
                    extent=[0, 2*np.pi, 0, 2*np.pi])  # 0 is vorticity

    plt.colorbar(cax, ax=ax)

    tick_positions = np.linspace(0, 2*np.pi, 3)  # Creates ticks at 0, π, 2π
    tick_labels = [r"$0$", r"$\pi$", r"$2\pi$"]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    # Set title
    ax.set_title(f"$\omega$ at T= {timestep} x{time_params.save_int} dt")

    return fig, ax  # Return the figure and axis objects