"""Constants for dataset soccer-net-v2"""

from SoccerNet.utils import getListGames
from pathlib import Path

work_dir = Path("/usr/users/siapartnerscomsportif/bohin_ant/conv-model/")
data_dir = work_dir / "data"
configs_dir = work_dir / "configs"
soccernet_dir = data_dir / "soccernet"

action_dir = data_dir / "action"
configs_dir = configs_dir / "action"
experiments_dir = action_dir / "experiments"
predictions_dir = action_dir / "predictions"
visualizations_dir = action_dir / "visualizations"

soccernet_dir = soccernet_dir / "action-spotting-2023"

"""
val_games = [
    'england_epl/2015-2016/2016-01-23 - 20-30 West Ham 2 - 2 Manchester City',
    'england_epl/2016-2017/2016-10-01 - 14-30 Swansea 1 - 2 Liverpool',
    'england_epl/2016-2017/2017-04-09 - 18-00 Everton 4 - 2 Leicester',
    'europe_uefa-champions-league/2014-2015/2014-11-05 - 22-45 Manchester City 1 - 2 CSKA Moscow',
    'europe_uefa-champions-league/2016-2017/2016-09-28 - 21-45 Napoli 4 - 2 Benfica',
    'europe_uefa-champions-league/2016-2017/2016-10-19 - 21-45 Paris SG 3 - 0 Basel',
    'france_ligue-1/2016-2017/2016-08-21 - 21-45 Paris SG 3 - 0 Metz',
    'france_ligue-1/2016-2017/2016-09-09 - 21-45 Paris SG 1 - 1 St Etienne',
    'france_ligue-1/2016-2017/2017-04-09 - 22-00 Paris SG 4 - 0 Guingamp',
    'germany_bundesliga/2015-2016/2015-10-04 - 18-30 Bayern Munich 5 - 1 Dortmund',
    'germany_bundesliga/2016-2017/2016-12-03 - 17-30 Dortmund 4 - 1 B. Monchengladbach',
    'germany_bundesliga/2016-2017/2017-02-25 - 17-30 SC Freiburg 0 - 3 Dortmund',
    'italy_serie-a/2016-2017/2016-08-20 - 19-00 AS Roma 4 - 0 Udinese',
    'italy_serie-a/2016-2017/2017-01-22 - 22-45 AS Roma 1 - 0 Cagliari',
    'italy_serie-a/2016-2017/2017-05-06 - 19-00 Napoli 3 - 1 Cagliari',
    'spain_laliga/2014-2015/2015-05-02 - 19-00 Atl. Madrid 0 - 0 Ath Bilbao',
    'spain_laliga/2016-2017/2016-08-21 - 21-15 Real Sociedad 0 - 3 Real Madrid',
    'spain_laliga/2016-2017/2017-05-14 - 21-00 Las Palmas 1 - 4 Barcelona',
]
train_ignore_games = [
    'france_ligue-1/2016-2017/2017-05-14 - 22-00 St Etienne 0 - 5 Paris SG',
    'italy_serie-a/2016-2017/2016-08-28 - 21-45 Cagliari 2 - 2 AS Roma',
    'italy_serie-a/2016-2017/2016-09-16 - 21-45 Sampdoria 0 - 1 AC Milan',
    'italy_serie-a/2016-2017/2016-09-18 - 21-45 Fiorentina 1 - 0 AS Roma',
    'italy_serie-a/2016-2017/2016-09-21 - 21-45 AS Roma 4 - 0 Crotone',
]
train_games = sorted(
    set(
        getListGames(split="train", task="spotting", dataset="SoccerNet")
        + getListGames(split="valid", task="spotting", dataset="SoccerNet")
    )
    - set(val_games) - set(train_ignore_games)
)
"""
train_games = [
    'england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley',
    'england_epl/2014-2015/2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal',
    'england_epl/2014-2015/2015-02-21 - 18-00 Swansea 2 - 1 Manchester United',
    'england_epl/2014-2015/2015-02-22 - 19-15 Southampton 0 - 2 Liverpool',
    'england_epl/2015-2016/2015-08-08 - 19-30 Chelsea 2 - 2 Swansea',
    'england_epl/2015-2016/2015-08-29 - 17-00 Chelsea 1 - 2 Crystal Palace',
    'england_epl/2015-2016/2015-08-29 - 17-00 Manchester City 2 - 0 Watford',
    'england_epl/2015-2016/2015-09-12 - 14-45 Everton 3 - 1 Chelsea',
    'england_epl/2015-2016/2015-09-12 - 17-00 Crystal Palace 0 - 1 Manchester City',
    'england_epl/2015-2016/2015-09-19 - 19-30 Manchester City 1 - 2 West Ham',
    'england_epl/2015-2016/2015-09-26 - 17-00 Liverpool 3 - 2 Aston Villa',
    'england_epl/2015-2016/2015-10-17 - 17-00 Chelsea 2 - 0 Aston Villa',
    'england_epl/2015-2016/2015-10-31 - 15-45 Chelsea 1 - 3 Liverpool',
    'england_epl/2015-2016/2015-11-07 - 18-00 Manchester United 2 - 0 West Brom',
    'england_epl/2015-2016/2015-11-21 - 20-30 Manchester City 1 - 4 Liverpool',
    'england_epl/2015-2016/2015-11-29 - 15-00 Tottenham 0 - 0 Chelsea',
    'england_epl/2015-2016/2015-12-05 - 20-30 Chelsea 0 - 1 Bournemouth',
    'england_epl/2015-2016/2015-12-19 - 18-00 Chelsea 3 - 1 Sunderland',
    'england_epl/2015-2016/2015-12-26 - 18-00 Manchester City 4 - 1 Sunderland',
    'england_epl/2015-2016/2016-01-03 - 16-30 Crystal Palace 0 - 3 Chelsea',
    'england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom',
    'england_epl/2015-2016/2016-02-14 - 19-15 Manchester City 1 - 2 Tottenham'
]

val_games = [
    'england_epl/2015-2016/2016-02-07 - 19-00 Chelsea 1 - 1 Manchester United'
]

#test_games=[
#    'england_epl/2015-2016/2016-01-13 - 22-45 Chelsea 2 - 2 West Brom'
#]

test_games=[
    'custom'
]

#test_games = getListGames(split="test", task="spotting", dataset="SoccerNet")
challenge_games = getListGames(split="challenge", task="spotting", dataset="SoccerNet")
split2games = {
    "train": train_games,
    "val": val_games,
    "test": test_games,
    "challenge": challenge_games,
}

classes = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Card",
]

card_classes = [
    "Yellow card",
    "Red card",
    "Yellow->red card",
]

num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

num_halves = 1
halves = list(range(1, num_halves + 1))
postprocess_params = {
    "gauss_sigma": 3.0,
    "height": 0.2,
    "distance": 15,
}

video_fps = 25.0