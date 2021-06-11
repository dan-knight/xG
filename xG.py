import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score

from plot import Pitch


class ExpectedGoalsModel:
    def __init__(self, shot_data):
        data = shot_data[shot_data['y'] > 50].copy()
        data['y'] = (data['y'] - 50) * 2
        data = self._calc_features(data)

        self.model, self.score = self._fit_model(data[['goal_angle', 'header']], data['goal'].astype(int))

    @staticmethod
    def _fit_model(x, y):
        model = linear_model.LogisticRegression().fit(x, y)
        score = cross_val_score(linear_model.LogisticRegression(), x, y, cv=10).mean()
        return model, score

    @staticmethod
    def _get_goal_angle(x, y):
        goal_size = 9.5
        goalposts = (50 + goal_size * 0.5, 50 - goal_size * 0.5)

        def get_dist(dx, dy):
            return np.sqrt(dx.pow(2) + (100 - dy).pow(2))

        a = goal_size
        b, c = get_dist(abs(x - goalposts[0]), y), get_dist(abs(x - goalposts[1]), y)
        return np.arccos((b.pow(2) + c.pow(2) - (a ** 2)) / (2 * b * c))

    @staticmethod
    def _get_distance(delta_x, y):
        return np.sqrt(delta_x.pow(2) + (100 - y).pow(2))

    @staticmethod
    def _get_radians(delta_x, y):
        return np.arctan((delta_x / y).to_numpy())

    @staticmethod
    def _calc_features(data):
        data['delta_x'] = (data['x'] - 50).abs()
        data['radians'] = ExpectedGoalsModel._get_radians(data['delta_x'], data['y']) / 1.58
        data['goal_angle'] = ExpectedGoalsModel._get_goal_angle(data['x'], data['y']) / 1.58
        data['distance'] = ExpectedGoalsModel._get_distance(data['delta_x'], data['y'])

        def scale_features():
            max_radians = 1.58

            data['goal_angle'] = data['goal_angle'] / max_radians
            data['radians'] = data['radians'] / max_radians
            data['distance'] = data['distance'] / data['distance'].max()

        scale_features()
        return data

    def predict(self, data):
        data = self._calc_features(data)
        return self.model.predict_proba(data[['goal_angle', 'header']])[:, 1].round(3)

    def plot(self, pitch, figure, header=False, limit=None):
        points = pd.concat([pd.DataFrame({'x': x, 'y': range(1, 101)}) for x in range(1, 101)], ignore_index=True)
        points['header'] = bool(header)
        points['xG'] = self.predict(points)
        values = points.set_index(['x', 'y'])['xG'].to_numpy().reshape(100, 100)

        heatmap = pitch.ax.imshow(values.transpose(), cmap='GnBu', vmax=0.3,
                                  extent=[-0.2, 76 + 0.2, 0, 116 / 2 + 0.2])

        fig.colorbar(heatmap)


if __name__ == '__main__':
    shot_events = pd.read_csv('data/shot_events.csv', index_col=0)
    xG_model = ExpectedGoalsModel(shot_events)

    fig, ax = plt.subplots()
    p = Pitch(ax, plot=False)
    xG_model.plot(p, fig)

    print(xG_model.score)
    plt.show()




