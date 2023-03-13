import os
import pathlib

print(pathlib.Path(__file__).parent.resolve())

data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'march-machine-learning-mania-2023')
curr_season = 2022
first_season = 2003
drop_seasons = [2020]
train_ratio = 0.8
box_stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']