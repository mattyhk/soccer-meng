import numpy as np
from peewee import *
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
import random

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

# Create feature and outcome set that includes games across different seasons
# seasons - an array of all the seasons to be included

def createFeatureSetsAllGames(seasons, simple = True):
  outcomes = []
  features = []

  for season in seasons:
    print str(season)
    # X, Y = createFeatureSet(season, simple)
    # X, Y = createSimpleXY(season)
    X, Y = createXY(season)
    outcomes = outcomes + Y
    features = features + X

  return (np.array(features), np.array(outcomes))


############################## Following Functions are Sub Optimal #############################
############################# See Functions Following Raw SQL Header ###########################

# Create feature data set for a given season
# If we only want games won/lost in regulation to be included, set param to be True
# Return as arrays of labled outcomes and corresponding feature vector
# season - a Model representing a database/season
def createFeatureSet(s, simple, wonInRegulation = True):
  season = seasonNames[s]
  outcomes = []
  features = []
  if wonInRegulation:
    # Only want games that were not drawn, and finished in regulation
    games = season.Game.select().where(season.Game.current_period == 2).where((season.Game.home_score_full_time - season.Game.away_score_full_time) != 0)
    for game in games:
      try:
        print game.id
        x, y = createFeatureVector(game, season, simple)
        outcomes.append(y)
        features.append(x)
      except:
        print 'Game ' + str(game.id) + ' raised exception'
        pass
  else:
    # Add all games to set
    games = season.Game.select()
    for game in games:
      try:
        x, y = createFeatureVector(game, season, simple)
        outcomes.append(y)
        features.append(x)
      except:
        print 'Game ' + str(game.id) + ' raised exception'
        pass

  ids = [g.id for g in games]
  # Save text file of features for current season, and text file of outcomes
  np.savetxt("Files/" + s + "-features.txt", np.array(features), fmt='%.5f', delimiter=',')
  np.savetxt("Files/" + s + "-outcomes.txt", np.array(outcomes), fmt='%d')
  np.savetxt("Files/" + s + "-games.txt", np.array(ids), fmt='%d')
  
  return (features, outcomes)

 # Helper function to create feature vectors
 # Pass in argument to determine if we want the simple or advanced feature vector
def createFeatureVector(game, season, simple):
  if simple:
    return createSimpleFeatureVector(game, season)

  else:
    return createAdvancedFeatureVector(game, season)


# Creates the simple feature vector for each game
# Pick Team A (team in question) by random to get mix of home/away teams
# Features (For Team A):
# Total shots
# Total shots on goal
# Home/Away
# Total time of possession (as %)
# Total game time spent in opponents half (as %)
# Number of shots conceded
# Number of shots on goal conceded
#
# Also gets classification of the game - 1 if teamA wins, 0 otherwise
def createSimpleFeatureVector(game, season):
  home = 0
  shots = 0
  shotsOnGoal = 0
  possessionPercentage = 0
  attackingPercentage = 0
  shotsConceded = 0
  shotsOnGoalConceded = 0

  won = 0

  if random.random() >= 0.5:
    teamA = game.home_team
    teamB = game.away_team
    home = 1
    if game.home_score > game.away_score:
      won = 1
    possessionPercentage = game.possessions.get().home_team_percentage
    # Percentage of game time spent in opponent's half
    attackingPercentage = game.territories.get().away_team_percentage

  else:
    teamA = game.away_team
    teamB = game.home_team
    if game.home_score < game.away_score:
      won = 1
    possessionPercentage = game.possessions.get().away_team_percentage
    attackingPercentage = game.territories.get().home_team_percentage

  shots = game.shots.select().where(season.Shot.team == teamA).count()
  shotsOnGoal = game.shots.select().where(season.Shot.team == teamA).where(season.Shot.on_target == 1).count()
  shotsConceded = game.shots.select().where(season.Shot.team == teamB).count()
  shotsOnGoalConceded = game.shots.select().where(season.Shot.team == teamB).where(season.Shot.on_target == 1).count()

  features = [shots, shotsOnGoal, possessionPercentage, attackingPercentage, 
              shotsConceded, shotsOnGoalConceded, home]

  return (features, won)


# Creates a feature vector with both non obvious variables and obvious variables
# Again pick a Team A at random
# Features (For Team A):
# Number of passes
# Number of passes allowed
# Differential in passes
# Average length of pass
# Average length of pass allowed
# Differential in length of passes
# Total number of crosses
# Total number of crosses allowed
# Differential in crosses
# Average distance of shot - to middle of goal
# Average distance of shot allowed - to middle of goal
# Differential in shot distance
# Percentage of shots assisted
# Allowed percentage of shots assisted
# Differential in assisted shots
# Number of free kick shots
# Number of free kick shots allowed
# Differential in free kick shots
# Average position of tackles - X/Y
# Average position of tackles allowed - X/Y
# Differential in tackle position - X/Y
# Number of shots
# Number of shots allowed
# Differential in shots
# Number of shots on target
# Number of shots on target allowed
# Differential in shots on target
# Home / Away
def createAdvancedFeatureVector(game, season):
  home = 0

  won = 0

  if random.random() >= 0.5:
    teamA = game.home_team
    teamB = game.away_team
    home = 1
    if game.home_score > game.away_score:
      won = 1

  else:
    teamA = game.away_team
    teamB = game.home_team
    if game.away_score > game.home_score:
      won = 1

  numPasses, numCrosses, avgPassLength = passFeatures(game, season, teamA)
  numPassesAllowed, numCrossesAllowed, avgPassLengthAllowed = passFeatures(game, season, teamB)
  passDiff = numPasses - numPassesAllowed
  crossDiff = numCrosses - numCrossesAllowed
  passLengthDiff = avgPassLength - avgPassLengthAllowed

  numShots, freeKickShots, shotsOnTarget, assistedShotPercentage, avgShotDistance = shotFeatures(game, season, teamA)
  numShotsAllowed, freeKickShotsAllowed, shotsOnTargetAllowed, assistedShotPercentageAllowed, avgShotDistanceAllowed = shotFeatures(game, season, teamB)
  shotsDiff = numShots - numShotsAllowed
  shotsOnTargetDiff = shotsOnTarget - shotsOnTargetAllowed
  freeKicksDiff = freeKickShots - freeKickShotsAllowed
  assistedShotDiff = assistedShotPercentage - assistedShotPercentageAllowed
  shotDistanceDiff = avgShotDistance - avgShotDistanceAllowed

  avgXTacklePosition, avgYTacklePosition = tackleFeatures(game, season, teamA)
  avgXTacklePositionOpp, avgYTacklePositionOpp = tackleFeatures(game, season, teamB)
  tackleXDiff = avgXTacklePosition - avgXTacklePositionOpp
  tackleYDiff = avgYTacklePosition - avgYTacklePositionOpp

  features = [numPasses, numPassesAllowed, passDiff, avgPassLength, avgPassLengthAllowed, passLengthDiff, numCrosses, numCrossesAllowed, crossDiff, 
              avgShotDistance, avgShotDistanceAllowed, shotDistanceDiff, assistedShotPercentage, assistedShotPercentageAllowed, assistedShotDiff, freeKickShots, freeKickShotsAllowed, freeKicksDiff,
                avgXTacklePosition, avgYTacklePosition, avgXTacklePositionOpp, avgYTacklePositionOpp, tackleXDiff, tackleYDiff,
                  numShots, numShotsAllowed, shotsDiff, shotsOnTarget, shotsOnTargetAllowed, shotsOnTargetDiff, home]

  return (features, won)

# Function that creates the passing features
def passFeatures(season):
  passCounts = season.Game.select(Game, fn.Count()).join(season.Pass).join(season.Team)


# Function that returns all of the passing features desired
# Team specific
def passFeatures(game, season, team):
  passLength = 0.0
  passes = 0.0
  crosses = 0

  for p in team.passes.select().join(season.Game).where(season.Game.id == game.id):
    try:
      passLength += calcDistancePass(p)
      crosses += p.crosses
      passes += 1.0
    except:
      print "pass " + str(p.id) + " raised exception"
      pass

  avgPassLength = passLength / passes

  return passes, crosses, avgPassLength


# Function that returns all of the shooting features
# Team specific
def shotFeatures(game, season, team):
  freeKicks = 0.0
  shots = 0.0
  assistedShots = 0.0
  shotDistance = 0.0
  onTarget = 0.0

  for s in team.shots.select().join(season.Game).where(season.Game.id == game.id):
    shots += 1.0
    assistedShots += s.assisted
    freeKicks += max(s.direct_free_kick, s.free_kick)
    shotDistance += calcDistanceShot(s)
    onTarget += s.on_target

  assistPercent = assistedShots / shots
  avgShotDistance = shotDistance / shots

  return shots, freeKicks, onTarget, assistPercent, avgShotDistance

# Function that returns all of the tackling features
# Team specific
def tackleFeatures(game, season, team):
  totalTackles = 0.0
  xPositionTackle = 0.0
  yPositionTackle = 0.0

  for t in team.tackles.select().join(season.Game).where(season.Game.id == game.id):
    totalTackles += 1.0
    xPositionTackle += float(t.x)
    yPositionTackle += float(t.y)

  avgX = xPositionTackle / totalTackles
  avgY = yPositionTackle / totalTackles
  
  return avgX, avgY

# Calculates the distance of a shot from the target goal
# If its a goal, shot is missing end x and end y
# TO BE IMPLEMENTED
def calcDistanceShot(shot):
  distance = (((100.0 - float(shot.start_x)) ** 2) + ((50.0 - float(shot.start_y)) ** 2)) ** 0.5
  return distance


# Calculates the distance of a pass between two players
# May include passes that went out of play, do not have end x or end y
def calcDistancePass(p):
  distance = (((float(p.end_x) - float(p.start_x)) ** 2) + ((float(p.end_y) - float(p.start_y)) ** 2)) ** 0.5
  return distance


################################################# Raw Sql Queries ###########################################################

# Creates a feature array for pass metrics for the entire field for either the home or away team across an entire season
# Returns an array where each element is a tuple
# Tuple consists of:
# Game ID
# Team ID
# Total Passes
# Average Pass Length
# Number of Crosses
# Pass success rate
def totalPassMetrics(s, home = True):
  if home:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Pass.id),
                      fn.Avg(fn.Sqrt(fn.Power(s.Pass.start_x - s.Pass.end_x, 2) + fn.Power(s.Pass.start_y - s.Pass.end_y, 2))),
                      fn.Sum(s.Pass.crosses),
                      fn.Sum(s.Pass.outcome) / fn.count(s.Pass.id))
              .join(s.Pass)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Pass.team == s.Game.home_team))
              .group_by(s.Game.id)
              .tuples())
  else:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Pass.id),
                      fn.Avg(fn.Sqrt(fn.Power(s.Pass.start_x - s.Pass.end_x, 2) + fn.Power(s.Pass.start_y - s.Pass.end_y, 2))),
                      fn.Sum(s.Pass.crosses),
                      fn.Sum(s.Pass.outcome) / fn.count(s.Pass.id))
              .join(s.Pass)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Pass.team == s.Game.away_team))
              .group_by(s.Game.id)
              .tuples())

  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Creates a feature array for passing metrics in the attacking third for either the home or away team across an entire season
# Returns an array where each element is a tuple of features
# Tuple consists of:
# Game ID
# Team ID
# Total Passes
# Average Pass Length
# Pass Success Rate
def attackingPassMetrics(s, home = True):
  MINX = 66.6
  if home:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Pass.id),
                      fn.Avg(fn.Sqrt(fn.Power(s.Pass.start_x - s.Pass.end_x, 2) + fn.Power(s.Pass.start_y - s.Pass.end_y, 2))),
                      fn.Sum(s.Pass.outcome) / fn.count(s.Pass.id))
              .join(s.Pass)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Pass.team == s.Game.home_team)
                      & (s.Pass.start_x > MINX))
              .group_by(s.Game.id)
              .tuples())
  else:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Pass.id),
                      fn.Avg(fn.Sqrt(fn.Power(s.Pass.start_x - s.Pass.end_x, 2) + fn.Power(s.Pass.start_y - s.Pass.end_y, 2))),
                      fn.Sum(s.Pass.outcome) / fn.count(s.Pass.id))
              .join(s.Pass)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Pass.team == s.Game.away_team)
                      & (s.Pass.start_x > MINX))
              .group_by(s.Game.id)
              .tuples())
  
  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Creates a feature array for shot metrics for either the home or away team across an entire season
# Returns an array where each element is a tuple of features
# Tuple consists of:
# Game ID
# Team ID
# Total Shots
# Total Shots on Target
# Average Shot Distance
def totalShotMetrics(s, home = True):
  if home:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Shot.id),
                      fn.Sum(s.Shot.on_target),
                      fn.Avg(fn.Sqrt(fn.Power(100.0 - s.Shot.start_x, 2) + fn.Power(50.0 - s.Shot.start_y, 2))))
              .join(s.Shot)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Shot.team == s.Game.home_team))
              .group_by(s.Game.id)
              .tuples())

  else:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Shot.id),
                      fn.Sum(s.Shot.on_target),
                      fn.Avg(fn.Sqrt(fn.Power(100.0 - s.Shot.start_x, 2) + fn.Power(50.0 - s.Shot.start_y, 2))))
              .join(s.Shot)
              .where((s.Game.current_period == 2) 
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Shot.team == s.Game.away_team))
              .group_by(s.Game.id)
              .tuples())

  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Creates a feature array for shot metrics in the attacking middle third of the field
# Returns an array where each element is a tuple of features
# Tuple consists of:
# Game ID
# Team ID
# Total Shots
def attackingShotMetrics(s, home = True):
  MINX = 66.6
  MINY = 33.3
  MAXY = 66.6
  if home:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Shot.id))
              .join(s.Shot)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Shot.team == s.Game.home_team)
                      & (s.Shot.start_x > MINX)
                      & (s.Shot.start_y > MINY)
                      & (s.Shot.start_y < MAXY))
              .group_by(s.Game.id)
              .tuples())
  else:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Shot.id))
              .join(s.Shot)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Shot.team == s.Game.away_team)
                      & (s.Shot.start_x > MINX)
                      & (s.Shot.start_y > MINY)
                      & (s.Shot.start_y < MAXY))
              .group_by(s.Game.id)
              .tuples())

  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Creates a feature array for tackle metrics for the entire field for either the home or away team across an entire season
# Returns an array where each element is a tuple of features
# Tuple consists of:
# Game ID
# Team ID
# Average X Position of Tackle
# Average Y Position of Tackle
# Tackle Success Rate
def totalTackleMetrics(s, home=True):
  if home:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Avg(s.Tackle.x),
                      fn.Avg(s.Tackle.y),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2) 
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.home_team))
              .group_by(s.Game.id)
              .tuples())
  else:
    query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Avg(s.Tackle.x),
                      fn.Avg(s.Tackle.y),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2) 
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.away_team))
              .group_by(s.Game.id)
              .tuples())

  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Creates a feature array for tackle metrics in either the attacking or defensive third of the field
# Returns an array where each element is a tuple of features
# Tuple consists of:
# Game ID
# Team ID
# Number of Tackles
# Tackle Success Rate
def locationTackleMetrics(s, home = True, attacking = True):
  if attacking:
    MINX = 66.6
    if home:
      query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Tackle.id),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.home_team)
                      & (s.Tackle.x > MINX))
              .group_by(s.Game.id)
              .tuples())
    else:
      query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Tackle.id),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.away_team)
                      & (s.Tackle.x > MINX))
              .group_by(s.Game.id)
              .tuples())
  else:
    MAXX = 33.3
    if home:
      query = (s.Game
              .select(s.Game.id,
                      s.Game.home_team,
                      fn.Count(s.Tackle.id),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.home_team)
                      & (s.Tackle.x < MAXX))
              .group_by(s.Game.id)
              .tuples())
    else:
      query = (s.Game
              .select(s.Game.id,
                      s.Game.away_team,
                      fn.Count(s.Tackle.id),
                      fn.Sum(s.Tackle.outcome) / fn.Count(s.Tackle.id))
              .join(s.Tackle)
              .where((s.Game.current_period == 2)
                      & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0) 
                      & (s.Tackle.team == s.Game.away_team)
                      & (s.Tackle.x < MAXX))
              .group_by(s.Game.id)
              .tuples())

  features = {}

  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Create Feature Vector for Possessions over a Season
# Returns an array where each element is a tuple of features
# Tuple Consists of:
# Game ID
# Home Team ID
# Away Team ID
# Home Team Possession
# Away Team Possession
def possessionMetrics(s):
  query = (s.Game
            .select(s.Game.id,
                    s.Game.home_team,
                    s.Game.away_team,
                    s.Possession.home_team_percentage,
                    s.Possession.away_team_percentage)
            .join(s.Possession)
            .where((s.Game.current_period == 2)
                    & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0))
            .group_by(s.Game.id)
            .tuples())
  features = {}
  for q in query:
    gameID = q[0]
    features[gameID] = (q[1:])

  return features

# Create feature vector for only simple features over a season
# Also creates outcome vector
# Randomly chooses a team as Team A
# Features are:
# Number of Shots
# Number of Shots Allowed
# Number of Shots Diff
# Number of Shots on Target
# Number of Shots on Target Allowed
# Number of Shots on Target Diff
# Home / Away
# Possession Percentage
# Possession Percentage Allowed
# Possession Percentage Diff
def createSimpleXY(season):
  s = seasonNames[season]
  X = []
  y = []
  ids = []

  games = (s.Game.select(s.Game.id, s.Game.home_score_full_time, s.Game.away_score_full_time)
                .where((s.Game.current_period == 2) 
                        & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0))
                .tuples())

  homeTotalShots = totalShotMetrics(s)
  awayTotalShots = totalShotMetrics(s, False)
  possessions = possessionMetrics(s)

  for g in games:
    gID = g[0]
    ids.append(gID)

    if gID in possessions:
      possessionA, possessionB = possessions[gID][2:]
    else:
      possessionA, possessionB = [0.0, 0.0]

    if gID in homeTotalShots:
      shotsA, shotsOnTargetA, shotDistanceA = homeTotalShots[gID][1:]
    else:
      print str(gID) + " not in home total shots"
      shotsA, shotsOnTargetA, shotDistanceA = [0, 0, 0]

    if gID in awayTotalShots:
      shotsB, shotsOnTargetB, shotDistanceB = awayTotalShots[gID][1:]
    else:
      print str(gID) + " not in away total shots"
      shotsB, shotsOnTargetB, shotDistanceB = [0, 0, 0]

    shotDiff = shotsA - shotsB
    shotsOnTargetDiff = shotsOnTargetA - shotsOnTargetB
    possessionDiff = possessionA - possessionB

    if random.random() >= 0.5:
      home = 1
      if g[1] > g[2]:
        outcome = 1
      else:
        outcome = 0

      X.append([shotsA, shotsB, shotDiff, shotsOnTargetA, shotsOnTargetB, shotsOnTargetDiff,
                home, possessionA, possessionB, possessionDiff])

    else:
      home = 0
      if g[1] < g[2]:
        outcome = 1
      else:
        outcome = 0

      X.append([shotsB, shotsA, -(shotDiff), shotsOnTargetB, shotsOnTargetA, -(shotsOnTargetDiff),
                home, possessionB, possessionA, -(possessionDiff)])

    y.append(outcome)

  np.savetxt("Files/Model Simple" + season + "-features.txt", np.array(X), fmt='%.5f', delimiter=',')
  np.savetxt("Files/Model Simple" + season + "-outcomes.txt", np.array(y), fmt='%d')
  np.savetxt("Files/Model Simple" + season + "-games.txt", np.array(ids), fmt='%d')

  return X, y

# Create feature vector for all features over a season, and outcome vector
# Return two arrays: Outcomes, Features
# Randomly chooses the home or away team as Team A
def createXY(season):
  s = seasonNames[season]
  X = []
  y = []
  ids = []

  games = (s.Game.select(s.Game.id, s.Game.home_score_full_time, s.Game.away_score_full_time)
                .where((s.Game.current_period == 2) 
                        & ((s.Game.home_score_full_time - s.Game.away_score_full_time) != 0))
                .tuples())

  homeTotalPasses = totalPassMetrics(s)
  awayTotalPasses = totalPassMetrics(s, False)
  homeAttackingPasses = attackingPassMetrics(s)
  awayAttackingPasses = attackingPassMetrics(s, False)
  homeTotalShots = totalShotMetrics(s)
  awayTotalShots = totalShotMetrics(s, False)
  homeAttackingShots = attackingShotMetrics(s)
  awayAttackingShots = attackingShotMetrics(s, False)
  homeTotalTackles = totalTackleMetrics(s)
  awayTotalTackles = totalTackleMetrics(s, False)
  homeAttackingTackles = locationTackleMetrics(s)
  awayAttackingTackles = locationTackleMetrics(s, home = False)
  homeDefensiveTackles = locationTackleMetrics(s, home = True, attacking = False)
  awayDefensiveTackles = locationTackleMetrics(s, home = False, attacking = False)
  possessions = possessionMetrics(s)

  for g in games:
    gID = g[0]
    ids.append(gID)

    if gID in possessions:
      possessionA, possessionB = possessions[gID][2:]
    else:
      possessionA, possessionB = [0.0, 0.0]

    if gID in homeTotalPasses:
      passesA, avgPassLengthA, crossesA, passSuccessA = homeTotalPasses[gID][1:]
    else:
      print str(gID) + " not in home total passes"
      passesA, avgPassLengthA, crossesA, passSuccessA = [0, 0, 0, 0]

    if gID in awayTotalPasses:
      passesB, avgPassLengthB, crossesB, passSuccessB = awayTotalPasses[gID][1:]
    else:
      print str(gID) + " not in away total passes"
      passesB, avgPassLengthB, crossesB, passSuccessB = [0,0,0,0]

    if gID in homeAttackingPasses:
      attackingPassesA, attackingAvgPassLengthA, attackingPassSuccessA = homeAttackingPasses[gID][1:]
    else:
      print str(gID) + " not in home attacking passes"
      attackingPassesA, attackingAvgPassLengthA, attackingPassSuccessA = [0, 0, 0]

    if gID in awayAttackingPasses:
      attackingPassesB, attackingAvgPassLengthB, attackingPassSuccessB = awayAttackingPasses[gID][1:]
    else:
      print str(gID) + " not in away attack passes"
      attackingPassesB, attackingAvgPassLengthB, attackingPassSuccessB = [0, 0, 0]

    passDiff = passesA - passesB
    avgPassLengthDiff = avgPassLengthA - avgPassLengthB
    crossesDiff = crossesA - crossesB
    passSuccessDiff = passSuccessA - passSuccessB
    attackingPassDiff = attackingPassesA - attackingPassesB
    attackingPassLengthDiff = attackingAvgPassLengthA - attackingAvgPassLengthB
    attackingPassSuccessDiff = attackingPassSuccessA - attackingPassSuccessB
    possessionDiff = possessionA - possessionB

    if gID in homeTotalShots:
      shotsA, shotsOnTargetA, shotDistanceA = homeTotalShots[gID][1:]
    else:
      print str(gID) + " not in home total shots"
      shotsA, shotsOnTargetA, shotDistanceA = [0, 0, 0]

    if gID in awayTotalShots:
      shotsB, shotsOnTargetB, shotDistanceB = awayTotalShots[gID][1:]
    else:
      print str(gID) + " not in away total shots"
      shotsB, shotsOnTargetB, shotDistanceB = [0, 0, 0]

    if gID in homeAttackingShots:
      attackingShotsA = homeAttackingShots[gID][1]
    else:
      print str(gID) + " not in home attack shots"
      attackingShotsA = 0

    if gID in awayAttackingShots:
      attackingShotsB = awayAttackingShots[gID][1]
    else:
      print str(gID) + " not in away attack shots"
      attackingShotsB = 0

    shotDiff = shotsA - shotsB
    shotsOnTargetDiff = shotsOnTargetA - shotsOnTargetB
    shotDistanceDiff = shotDistanceA - shotDistanceB
    attackingShotsDiff = attackingShotsA - attackingShotsB

    if gID in homeTotalTackles:
      avgXTackleA, avgYTackleA, tackleSuccessA = homeTotalTackles[gID][1:]
    else:
      print str(gID) + ' not in home total tackles'
      avgXTackleA, avgYTackleA, tackleSuccessA = [0, 0, 0]

    if gID in awayTotalTackles:
      avgXTackleB, avgYTackleB, tackleSuccessB = awayTotalTackles[gID][1:]
    else:
      print str(gID) + ' not in away total tackles'
      avgXTackleB, avgYTackleB, tackleSuccessB = [0, 0, 0]

    if gID in homeAttackingTackles:
      attackingTacklesA, attackingTackleSuccessA = homeAttackingTackles[gID][1:]
    else:
      print str(gID) + ' not in home attacking tackles'
      attackingTacklesA, attackingTackleSuccessA = [0, 0]

    if gID in awayAttackingTackles:
      attackingTacklesB, attackingTackleSuccessB = awayAttackingTackles[gID][1:]
    else:
      print str(gID) + ' not in away attacking tackles'
      attackingTacklesB, attackingTackleSuccessB = [0, 0]

    if gID in homeDefensiveTackles:
      defensiveTacklesA, defensiveTackleSuccessA = homeDefensiveTackles[gID][1:]
    else:
      print str(gID) + ' not in home defensive tackles'
      defensiveTacklesA, defensiveTackleSuccessA = [0, 0]

    if gID in awayDefensiveTackles:
      defensiveTacklesB, defensiveTackleSuccessB = awayDefensiveTackles[gID][1:]
    else:
      print str(gID) + ' not in away defensive tackles'
      defensiveTacklesB, defensiveTackleSuccessB = [0, 0]

    avgXTackleDiff = avgXTackleA - avgXTackleB
    avgYTackleDiff = avgYTackleA - avgYTackleB
    tackleSuccessDiff = tackleSuccessA - tackleSuccessB
    attackingTacklesDiff = attackingTacklesA - attackingTacklesB
    attackingTackleSuccessDiff = attackingTackleSuccessA - attackingTackleSuccessB
    defensiveTacklesDiff = defensiveTacklesA - defensiveTacklesB
    defensiveTackleSuccessDiff = defensiveTackleSuccessA - defensiveTackleSuccessB

    if random.random() >= 0.5:
      home = 1
      if g[1] > g[2]:
        outcome = 1
      else:
        outcome = 0

      # X.append([passesA, passesB, passDiff, avgPassLengthA, avgPassLengthB, avgPassLengthDiff, crossesA, crossesB, crossesDiff, passSuccessA, passSuccessB, passSuccessDiff,
      #           attackingPassesA, attackingPassesB, attackingPassDiff, attackingAvgPassLengthA, attackingAvgPassLengthB, attackingPassLengthDiff, attackingPassSuccessA, attackingPassSuccessB, attackingPassSuccessDiff,
      #           shotDistanceA, shotDistanceB, shotDistanceDiff, shotsA, shotsB, shotDiff, shotsOnTargetA, shotsOnTargetB, shotsOnTargetDiff,
      #           attackingShotsA, attackingShotsB, attackingShotsDiff,
      #           avgXTackleA, avgXTackleB, avgYTackleA, avgYTackleB, avgXTackleDiff, avgYTackleDiff, tackleSuccessA, tackleSuccessB, tackleSuccessDiff,
      #           attackingTacklesA, attackingTacklesB, attackingTacklesDiff, attackingTackleSuccessA, attackingTackleSuccessB, attackingTackleSuccessDiff,
      #           defensiveTacklesA, defensiveTacklesB, defensiveTacklesDiff, defensiveTackleSuccessA, defensiveTackleSuccessB, defensiveTackleSuccessDiff,
      #           home, possessionA, possessionB, possessionDiff])

      X.append([passesA, passesB, passDiff, avgPassLengthA, avgPassLengthB, avgPassLengthDiff, crossesA, crossesB, crossesDiff, passSuccessA, passSuccessB, passSuccessDiff,
                attackingPassesA, attackingPassesB, attackingPassDiff, attackingAvgPassLengthA, attackingAvgPassLengthB, attackingPassLengthDiff, attackingPassSuccessA, attackingPassSuccessB, attackingPassSuccessDiff,
                shotDistanceA, shotDistanceB, shotDistanceDiff, shotsA, shotsB, shotDiff, shotsOnTargetA, shotsOnTargetB, shotsOnTargetDiff,
                attackingShotsA, attackingShotsB, attackingShotsDiff,
                avgXTackleA, avgXTackleB, avgYTackleA, avgYTackleB, avgXTackleDiff, avgYTackleDiff, tackleSuccessA, tackleSuccessB, tackleSuccessDiff,
                attackingTacklesA, attackingTacklesB, attackingTacklesDiff, attackingTackleSuccessA, attackingTackleSuccessB, attackingTackleSuccessDiff,
                defensiveTacklesA, defensiveTacklesB, defensiveTacklesDiff, defensiveTackleSuccessA, defensiveTackleSuccessB, defensiveTackleSuccessDiff,
                home])

    else:
      home = 0
      if g[1] < g[2]:
        outcome = 1
      else:
        outcome = 0

      # X.append([passesB, passesA, -(passDiff), avgPassLengthB, avgPassLengthA, -(avgPassLengthDiff), crossesB, crossesA, -(crossesDiff), passSuccessB, passSuccessA, -(passSuccessDiff),
      #           attackingPassesB, attackingPassesA, -(attackingPassDiff), attackingAvgPassLengthB, attackingAvgPassLengthA, -(attackingPassLengthDiff), attackingPassSuccessB, attackingPassSuccessA, -(attackingPassSuccessDiff),
      #           shotDistanceB, shotDistanceA, -(shotDistanceDiff), shotsB, shotsA, -(shotDiff), shotsOnTargetB, shotsOnTargetA, -(shotsOnTargetDiff),
      #           attackingShotsB, attackingShotsA, -(attackingShotsDiff),
      #           avgXTackleB, avgXTackleA, avgYTackleB, avgYTackleA, -(avgXTackleDiff), -(avgYTackleDiff), tackleSuccessB, tackleSuccessA, -(tackleSuccessDiff),
      #           attackingTacklesB, attackingTacklesA, -(attackingTacklesDiff), attackingTackleSuccessB, attackingTackleSuccessA, -(attackingTackleSuccessDiff),
      #           defensiveTacklesB, defensiveTacklesA, -(defensiveTacklesDiff), defensiveTackleSuccessB, defensiveTackleSuccessA, -(defensiveTackleSuccessDiff),
      #           home, possessionB, possessionA, -(possessionDiff)])

      X.append([passesB, passesA, -(passDiff), avgPassLengthB, avgPassLengthA, -(avgPassLengthDiff), crossesB, crossesA, -(crossesDiff), passSuccessB, passSuccessA, -(passSuccessDiff),
                attackingPassesB, attackingPassesA, -(attackingPassDiff), attackingAvgPassLengthB, attackingAvgPassLengthA, -(attackingPassLengthDiff), attackingPassSuccessB, attackingPassSuccessA, -(attackingPassSuccessDiff),
                shotDistanceB, shotDistanceA, -(shotDistanceDiff), shotsB, shotsA, -(shotDiff), shotsOnTargetB, shotsOnTargetA, -(shotsOnTargetDiff),
                attackingShotsB, attackingShotsA, -(attackingShotsDiff),
                avgXTackleB, avgXTackleA, avgYTackleB, avgYTackleA, -(avgXTackleDiff), -(avgYTackleDiff), tackleSuccessB, tackleSuccessA, -(tackleSuccessDiff),
                attackingTacklesB, attackingTacklesA, -(attackingTacklesDiff), attackingTackleSuccessB, attackingTackleSuccessA, -(attackingTackleSuccessDiff),
                defensiveTacklesB, defensiveTacklesA, -(defensiveTacklesDiff), defensiveTackleSuccessB, defensiveTackleSuccessA, -(defensiveTackleSuccessDiff),
                home])

    y.append(outcome)

  np.savetxt("Files/Model 4/" + season + "-features.txt", np.array(X), fmt='%.5f', delimiter=',')
  np.savetxt("Files/Model 4/" + season + "-outcomes.txt", np.array(y), fmt='%d')
  np.savetxt("Files/Model 4/" + season + "-games.txt", np.array(ids), fmt='%d')

  return X, y



######################################## Loading data from txt Files ######################################

# Function that creates an array from different files
# Each file represents an array
def readArray(files, d=None):
  seq = []
  for f in files:
    seq.append(np.loadtxt(f, delimiter=d))
  return np.concatenate(seq)

# Creates a sequence of filenames for the given seasons
# File name depends on given ending
def createFileNames(seasons, ending, model):
  names = []
  for s in seasons:
    # f = '/Users/Files/' + model + '/' + s + '-' + ending + '.txt'
    f = '/Users/Matthew/Dropbox (MIT)/MENG/Football/python/Files/' + model + '/' + s + '-' + ending + '.txt'
    names.append(f)
  return names

# Need to delete repeated features - number 13 and 14
# - Done
def deleteColumns(seasons):
  files = createFileNames(seasons, "features")
  for f in files:
    arr = np.loadtxt(f, delimiter=',')
    arr = np.delete(arr, [12,13], axis=1)
    np.savetxt(f, arr, fmt='%.5f', delimiter=',')

# Return X, y array of featuers and outcomes from txt files
def createXYFromFiles(seasons, model):
  features = createFileNames(seasons, "features", model)
  outcomes = createFileNames(seasons, "outcomes", model)
  x = readArray(features, d=',')
  y = readArray(outcomes)
  return x, y



################################# Machine Learning ########################################

# Initialises the classifier (Logistic Regression Model with L1 Regularisation)
# Fits weights based on the input data sets and error parameter C
# outcomes, features - data sets in question
# C - error parameter C, lower C results in sparser weights
def createClassifier(X, y, C = 1):
  clf = LogisticRegression(penalty = 'l1')
  clf = clf.fit(X, y)
  return clf

# Create a training/test set split
def createSplit(X, y, testsize = 0.2):
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = testsize)
  return X_train, X_test, y_train, y_test

# Method to choose best error parameter C
# The main goal of this exercise is to find features that are important
# We are looking to create sparse predictors, i.e. C should be low
# We will find C through a cross-validated grid search on the training data set
def findBestEstimator(X, y, l_2):
  params = [{'C': [0.001, 0.01, 0.1, 0.5, 1, 1.5, 10]}]
  if l_2:
    clf = GridSearchCV(LogisticRegression(penalty = 'l2'), params, cv = 5, scoring = 'accuracy')
    clf = clf.fit(X, y)
  else:
    clf = GridSearchCV(LogisticRegression(penalty = 'l1'), params, cv = 5, scoring = 'accuracy')
    clf = clf.fit(X, y)
  return clf.best_estimator_, clf.best_params_, clf

# Trains a Logistic Regression model on the data
# Takes in a full data set, and finds the best classifier through 
# k-fold cross validated Grid Search
# Produces a classification report, as well as other metrics
def trainBestModel(X, y, l_2):
  X_train, X_test, y_train, y_test = createSplit(X, y)
  X_means, X_stdDevs = findMeanStdDev(X_train)
  X_train_normalised = normaliseColumns(X_train, X_means, X_stdDevs)
  X_test_normalised = normaliseColumns(X_test, X_means, X_stdDevs)
  clf, params, grid = findBestEstimator(X_train_normalised, y_train, l_2)
  return clf, X_train_normalised, X_test_normalised, y_train, y_test

# Trains a sparse model at C=0.01
# Prints all the necessary metrics
def trainSparseModel(X, y, c=0.01):
  clf = LogisticRegression(penalty='l1', C=c)
  clf = clf.fit(X, y)
  return clf

# Prints the metrics for a given classifier and data set
def printScores(clf, X, y, heading):
  print "-------------" + heading + "-------------"
  y_predict = clf.predict(X)
  print "-------------Classification report-------------"
  print metrics.classification_report(y, y_predict)
  print "-------------Confusion Matrix-------------"
  print metrics.confusion_matrix(y, y_predict)
  print "-------------ROC AUC Score-------------"
  print metrics.roc_auc_score(y, y_predict)
  print "-------------Accuracy-------------"
  print clf.score(X, y)


################################# Preprocessing #################################

# Normalises the columns of a matrix by using a z-score
# Takes the Matrix, Means and Standard Devs for each Column
# y' = (y - m) / sigma
def normaliseColumns(M, means, stdDevs):
  A = M - means
  B = A / stdDevs
  return B

# Finds the Means and Standard Devs for each column of a matrix M
def findMeanStdDev(M):
  means = np.mean(M, axis=0)
  stdDevs = np.std(M, axis=0)
  return means, stdDevs



################################# Main #################################

def run(model, l_2 = True):
  X, y = createXYFromFiles(seasons, model)
  clf, X_train, X_test, y_train, y_test = trainBestModel(X, y, l_2)
  printScores(clf, X_test, y_test, "Test Set")
  print '--------------Coefficients Are:----------------'
  for c in clf.coef_[0]:
    print c
  return clf, X_train, X_test, y_train, y_test


# Removing shots on target as a predictor
# Columns 27, 28, 29 (0-based)
# Model 5
def run_no_shots_target(model, l_2 = True):
  X, y = createXYFromFiles(seasons, model)
  X = np.delete(X, [27, 28, 29], axis = 1)
  clf, X_train, X_test, y_train, y_test = trainBestModel(X, y, l_2)
  printScores(clf, X_test, y_test, "Test Set")
  print '--------------Coefficients Are:----------------'
  for c in clf.coef_[0]:
    print c
  return clf, X_train, X_test, y_train, y_test

# Run the same as above
# Remove shots on target, and shots in general
# Use model 5, and remove the columns 24, 25, 26, 27, 28, 29
def run_non_obvious(model = 'Model 5', l_2 = True):
  X, y = createXYFromFiles(seasons, model)
  X = np.delete(X, [24, 25, 26, 27, 28, 29, 54, 55, 56, 57], axis = 1)
  clf, X_train, X_test, y_train, y_test = trainBestModel(X, y, l_2)
  printScores(clf, X_test, y_test, "Test Set")
  print '--------------Coefficients Are:----------------'
  for c in clf.coef_[0]:
    print c
  return clf, X_train, X_test, y_train, y_test

def run_baseline_home_team(model = 'Model 5'):
  X, y = createXYFromFiles(seasons, model)
  X = X[:, 54]
  X_train, X_test, y_train, y_test = createSplit(X, y)
  y_predict = X_test == 1
  y_predict = y_predict.astype('float')
  print metrics.accuracy_score(y_test, y_predict)
























