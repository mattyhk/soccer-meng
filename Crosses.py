# Looking at Crosses More in Depth
import numpy as np
import random
from peewee import *

import Models.aleague1314_models as algue13
import Models.bundesliga1213_models as bnds12
import Models.bundesliga1314_models as bnds13
import Models.epl1011_models as epl10
import Models.epl1112_models as epl11
import Models.epl1213_models as epl12
import Models.epl1314_models as epl13
import Models.europa1213_models as erp12
import Models.laliga1213_models as liga12
import Models.laliga1314_models as liga13
import Models.ligue11213_models as lgue12
import Models.ligue11314_models as lgue13
import Models.serieA1213_models as ser12
import Models.serieA1314_models as ser13
import Models.ucl1011_models as ucl10
import Models.ucl1112_models as ucl11
import Models.ucl1213_models as ucl12
import Models.ucl1314_models as ucl13
import Models.wcup_models as wc

# Array of all the databases/seasons in our data set
# Each database is represented by a Model as defined by the peewee module
seasons = ["algue13", "bnds12", "bnds13", "epl10", "epl11", "epl12", "epl13",
              "erp12", "liga12", "liga13", "lgue12", "lgue13", "ser12", "ser13", "ucl10", "ucl11", "ucl12", "ucl13", "wc"]

seasonNames = {"algue13": algue13, "bnds12": bnds12, "bnds13": bnds13, "epl10": epl10, "epl11": epl11, "epl12": epl12, "epl13": epl13, "erp12": erp12, "liga12": liga12,
                "liga13": liga13, "lgue12": lgue12, "lgue13": lgue13, "ser12": ser12, "ser13": ser13, "ucl10": ucl10, "ucl11": ucl11, "ucl12": ucl12, "ucl13": ucl13, "wc": wc }

####################################### Investigation ##############################################

# In order to compensate for other game features,
# randomly pick a team for each game to be the team in question
# Creates a dataset such that for each game there is one team, and the goals scored and allowed
# are adjusted accordingly
def randomiseTeam(goals, crosses):
  goalsPerGame = []
  crossesPerGame = []

  for i in range(len(goals)):

    # Choose home team
    if random.random() >= 0.5:
      crossesPerGame.append(crosses[i][0])
      goalsScored = goals[i][0]
      goalsAllowed = goals[i][1]
    
    else:
      crossesPerGame.append(crosses[i][1])
      goalsScored = goals[i][1]
      goalsAllowed = goals[i][0]

    goalsPerGame.append([goalsScored, goalsAllowed])

  return crossesPerGame, goalsPerGame


# Find the correlation betweem the number of crosses and goals scored and allowed
# goals - list where each element is (goalsScored, goalsAllowed)
def findCorrCrossGoals(goals, crosses):
  goalsScored = goals[:, 0]
  goalsAllowed = goals[:, 1]
  # print "Goals Scored Correlation"
  # print np.corrcoef(crosses, goalsScored)
  # print "Goals Allowed Correlation"
  # print np.corrcoef(crosses, goalsAllowed)
  return np.corrcoef(crosses, goalsScored), np.corrcoef(crosses, goalsAllowed)


##################################### Creating Dataset #############################################
# Retrieves data from database
def getData(seasons):
  allGoals = []
  allCrosses = []
  for s in seasons:
    print s
    goals, gameID = totalGoals(s)
    crosses = totalCrosses(s, gameID)
    allGoals = allGoals + goals
    allCrosses = allCrosses + crosses

  return allGoals, allCrosses

# Retrieves data from txt files
def readData(files, d = ','):
  seq = []
  for f in files:
    seq.append(np.loadtxt(f, delimiter = d))
  return np.concatenate(seq)

# Creates a sequence of appropriate file names
def createFileNames(seasons, ending):
  names = []
  for s in seasons:
    # f = 'Files/Crosses/' + s + '-' + ending + '.txt'
    f = '/Users/Matthew/Dropbox (MIT)/MENG/Football/python/Files/Crosses/' + s + '-' + ending + '.txt'
    names.append(f)
  return names

# Creates lists of goals and crosses
def createGoalsCrosses(seasons):
  goalFiles = createFileNames(seasons, 'goals')
  crossFiles = createFileNames(seasons, 'crosses')
  goals = readData(goalFiles)
  crosses = readData(crossFiles)
  goals = goals[:, 3:]
  crosses = crosses[:, 1:]
  return goals, crosses


##################################### Database Queries #############################################

# Return all the home and away goals scored per game over a season
def totalGoals(season):
  s = seasonNames[season]
  gameID = []

  query = (s.Game
            .select(s.Game.id,
                    s.Game.home_team,
                    s.Game.away_team,
                    s.Game.home_score_full_time,
                    s.Game.away_score_full_time)
            .where((s.Game.current_period == 2))
            .tuples())

  goals = [q for q in query]
  gameID = [q[0] for q in query]

  np.savetxt("Files/Crosses/" + season + "-goals.txt", np.array(goals), fmt = '%d', delimiter = ',')
  np.savetxt("Files/Crosses/" + season + "-games.txt", np.array(gameID), fmt = '%d')

  return goals, gameID

# Return the number of crosses for both the home and away team, as well as the difference
# For an entire season
def totalCrosses(season, games):
  s = seasonNames[season]

  homeQuery = (s.Game
                .select(s.Game.id,
                        fn.Sum(s.Pass.crosses))
                .join(s.Pass)
                .where((s.Game.current_period == 2)
                        & (s.Pass.team == s.Game.home_team))
                .group_by(s.Game.id)
                .tuples())

  awayQuery = (s.Game
                .select(s.Game.id,
                        fn.Sum(s.Pass.crosses))
                .join(s.Pass)
                .where((s.Game.current_period == 2)
                        & (s.Pass.team == s.Game.away_team))
                .group_by(s.Game.id)
                .tuples())

  crosses = []

  j = 0
  k = 0

  # Need to ensure the goals and crosses lists match on a per game basis
  for i in range(len(games)):
    gameID = games[i]
    homeID = homeQuery[j][0]
    awayID = awayQuery[k][0]

    if gameID == homeID:
      homeCrosses = homeQuery[j][1]
      j += 1

    else:
      homeCrosses = 0.0

    if gameID == awayID:
      awayCrosses = awayQuery[k][1]
      k += 1

    else:
      awayCrosses = 0.0

    crossDiff = homeCrosses - awayCrosses

    crosses.append([gameID, homeCrosses, awayCrosses, crossDiff])

  np.savetxt("Files/Crosses/" + season + "-crosses.txt", np.array(crosses), fmt = '%d', delimiter = ',')

  return crosses


####################################### First Run Through ###########################################

def firstRun():
  goals, crosses = getData(seasons)
  goalsFiles, crossesFiles = createGoalsCrosses(seasons)

  goalsPerGame, crossesPerGame = randomiseTeam(goalsFiles, crossesFiles)
  findCorrCrossGoals(goalsPerGame, crossesPerGame)

def findCorrs():
  goalsFiles, crossesFiles = createGoalsCrosses(seasons)
  crossesPerGame, goalsPerGame = randomiseTeam(goalsFiles, crossesFiles)
  crossesPerGame = np.array(crossesPerGame)
  goalsPerGame = np.array(goalsPerGame)
  scored, allowed = findCorrCrossGoals(goalsPerGame, crossesPerGame)
  return scored, allowed


####################################### Significance Testing ###########################################

def meanBootstrap(seasons):
  corrValuesScored = []
  corrValuesAllowed = []

  goalsFiles, crossesFiles = createGoalsCrosses(seasons)

  for i in range(10000):
    crossesPerGame, goalsPerGame = randomiseTeam(goalsFiles, crossesFiles)
    crossesPerGame = np.array(crossesPerGame)
    goalsPerGame = np.array(goalsPerGame)
    scored, allowed = findCorrCrossGoals(goalsPerGame, crossesPerGame)
    corrScored = scored[0, 1]
    corrAllowed = allowed[0, 1]
    corrValuesScored.append(corrScored)
    corrValuesAllowed.append(corrAllowed)

  return corrValuesScored, corrValuesAllowed
















































