import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line


def lineplot(xs, ys_population, title, path='', xaxis='episode'):
    max_colour = 'rgb(0, 132, 180)'
    mean_colour = 'rgb(0, 172, 237)'
    std_colour = 'rgba(29, 202, 255, 0.2)'
    transparent = 'rgba(0, 0, 0, 0)'

    if isinstance(ys_population[0], (list, tuple)):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max = ys.min(1), ys.max(1)
        ys_mean, ys_std = ys.mean(1), ys.std(1)
        ys_median = np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        data = [
            Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False),
            Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean'),
            Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False),
            Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min'),
            Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max'),
            Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median'),
        ]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]

    plotly.offline.plot(
        {'data': data, 'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})},
        filename=os.path.join(path, title + '.html'),
        auto_open=False,
    )


def write_video(frames, title, path=''):
    frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[:, :, :, ::-1]
    _, H, W, _ = frames.shape
    writer = cv2.VideoWriter(
        os.path.join(path, '%s.mp4' % title),
        cv2.VideoWriter_fourcc(*'mp4v'),
        30.,
        (W, H),
        True,
    )
    for frame in frames:
        writer.write(frame)
    writer.release()
