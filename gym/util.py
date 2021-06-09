import matplotlib.pyplot as plt
import torch

def visualization(coords, tour_indices, title='None'):
    plt.close('all')

    num_plots = 3
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    axes = [a for ax in axes for a in ax]  # 2dim -> 1dim

    for i, ax in enumerate(axes):
        # idx 의 좌표 가져오기
        idx = tour_indices[i].unsqueeze(0)
        idx = idx.expand(2, -1)
        idx = torch.cat((idx, idx[:, 0:1]), dim=1)
        data = coords[i].transpose(1, 0)
        data = data.gather(1, idx).cpu().numpy()

        # draw graph
        ax.plot(data[0], data[1], zorder=1)
        ax.scatter(data[0], data[1], s=4, c='r', zorder=2)
        ax.scatter(data[0, 0], data[1, 0], s=20, c='k', marker='*', zorder=3)

        # limit 설정
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.title(title)
    plt.show()
