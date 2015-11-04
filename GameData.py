from BallEvents import *
from peewee import *
# from utils import *

import os
import numpy as np
import unicodedata


GAMES_DIR = 'Files/Thesis/'
TEAMS_DIR = 'Files/Thesis/'
folderNames = ['pass/', 'shot/', 'aerial/', 'clearance/', 'foul/', 'tackle/', 'takeon/', 'ballrecovery/', 'block/',
                'challenge/', 'dispossessed/', 'error/', 'interception/', 'keeperevent/', 'touch/', 'turnover/']
fileNames = ['pass', 'shot', 'aerial', 'clearance', 'foul', 'tackle', 'takeon', 'ballrecovery', 'block',
                'challenge', 'dispossessed', 'error', 'interception', 'keeperevent', 'touch', 'turnover']

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

# seasons = ["algue13", "bnds12", "bnds13", "epl10", "epl11", "epl12", "epl13",
#               "erp12", "liga12", "liga13", "lgue12", "lgue13", "ser12", "ser13", "ucl10", "ucl11", "ucl12", "ucl13", "wc"]
seasons = ["ucl10", "ucl11", "ucl12", "ucl13", "wc"]

# seasons = ["algue13", "epl13", "erp12", "liga12", "liga13", "lgue12", "lgue13", "ser12", "ser13", "ucl13", "wc"]

# thesisSeasons = ["algue13", "bnds12", "bnds13", "epl13", "erp12", "lgue12", "lgue13", "liga12", "liga13", "ser12", "ser13", "ucl13", "wc"]
thesisSeasons = ["bnds12", "bnds13", "epl13", "erp12", "lgue12", "lgue13", "liga12", "liga13", "ser12", "ser13", "ucl13", "wc"]

seasonNames = {"algue13": algue13, "bnds12": bnds12, "bnds13": bnds13, "epl10": epl10, "epl11": epl11, "epl12": epl12, "epl13": epl13, "erp12": erp12, "liga12": liga12,
                "liga13": liga13, "lgue12": lgue12, "lgue13": lgue13, "ser12": ser12, "ser13": ser13, "ucl10": ucl10, "ucl11": ucl11, "ucl12": ucl12, "ucl13": ucl13, "wc": wc }

eventNames = ["PASS", "SHOT", "AERIAL", "CLEARANCE", "FOUL", "TACKLE", "TAKE ON", "BALL RECOVERY", "BLOCK",
              "CHALLENGE", "DISPOSSESSED", "ERROR", "INTERCEPTION", "KEEPER EVENT", "TOUCH", "TURNOVER"]

eventClasses = [Pass, Shot, Aerial, Clearance, Foul, Tackle, TakeOn, BallRecovery, Block, Challenge, Dispossessed, Error, Interception, KeeperEvent, Touch, Turnover]

eventDict = {"PASS": Pass, "SHOT": Shot, "AERIAL": Aerial, "CLEARANCE": Clearance, "FOUL": Foul, "TACKLE": Tackle, "TAKE ON": TakeOn, "BALL RECOVERY": BallRecovery, "BLOCK": Block,
              "CHALLENGE": Challenge, "DISPOSSESSED": Dispossessed, "ERROR": Error, "INTERCEPTION": Interception, "KEEPER EVENT": KeeperEvent, "TOUCH": Touch, "TURNOVER": Turnover}

############################################ Reading / Writing Data ############################################################
def writeAllEvents(season):
  s = seasonNames[season]
  games = gameQuery(s)
  # eventQueries = [attackingMoveQuery, passQuery, shotQuery, aerialQuery, clearanceQuery, foulQuery, tackleQuery, takeOnQuery, 
  #                 ballRecoveryQuery, blockQuery, challengeQuery, dispossessedQuery, errorQuery, interceptionQuery, keeperQuery, 
  #                 touchQuery, turnoverQuery]
  eventQueries = [passQuery, shotQuery, aerialQuery, clearanceQuery, foulQuery, tackleQuery, takeOnQuery, 
                  ballRecoveryQuery, blockQuery, challengeQuery, dispossessedQuery, errorQuery, interceptionQuery, keeperQuery, 
                  touchQuery, turnoverQuery]
  events = {}

  print "Creating Events"

  for i in range(len(eventQueries)):
    q = eventQueries[i](s)
    # if i == 0:
    #   print "Creating Attacking Move Events"
    #   events[eventNames[i]] = createAttackingMoveEvents(q)
    if i == 0:
      print "Creating Pass Events"
      events[eventNames[i]] = createPassEvents(q)
    if i == 1:
      print "Creating Shot Events"
      events[eventNames[i]] = createShotEvents(q)
    if i > 1 and i <= 6:
      print "Creating Outcome Events"
      events[eventNames[i]] = createOutcomeEvents(q, eventClasses[i])
    if i > 6:
      print "Creating Events"
      events[eventNames[i]] = createEvents(q, eventClasses[i])

  gameEvents = splitEvents(games, events)
  del events

  print "Writing Events"

  for g in games:
    print "Writing " + str(g)
    e = sortEvents(gameEvents[g])
    writeEvents(g, season, e)

def filenameForGame(game, season = ''):
  path = os.path.abspath(str(game))
  if os.path.exists(path):
    return path

  path = GAMES_DIR + season + str(game)
  if os.path.exists(path):
    return path

  path = path + '.txt'
  if os.path.exists(path):
    return path

  # raise IOError("file not found for game: %s, %s" % (game, path))
  return path

def filenameForTeam(team):
  path = os.path.abspath(str(team))
  if os.path.exists(path):
    return path

  path = TEAMS_DIR + str(team)
  if os.path.exists(path):
    return path

  path = path + '.txt'
  if os.path.exists(path):
    return path
  return path

def dirNameForTeam(team):
  teamStr = str(team)
  path = os.path.abspath(teamStr)
  if os.path.exists(path):
    return path
  return TEAMS_DIR + teamStr

def makeTeamDir(team, s):
  teamStr = str(team)
  f = GAMES_DIR + s + '/' + teamStr + '/'
  if not os.path.exists(f):
    os.mkdir(f)
  return f

def getTeamDir(s, team):
  return GAMES_DIR + s + '/' + str(team) + '/'

def writeEventsToFile(events, f):
  arr = []
  for e in events:
    arr.append(e.toRow())
  np.savetxt(f, np.array(arr), fmt='%s', delimiter=',')

def writeEvents(game, season, events):
  name = filenameForGame(game, season)
  writeEventsToFile(events, name)

def writeTeamEvents(team, events):
  name = filenameForTeam(team)
  arr = [e.toRow() for e in events]
  np.savetxt(name, np.array(arr), fmt='%s', delimiter=',')

def writeTeamEventsForGameFile(team, events, gameFile):
  dirName = dirNameForTeam(team)
  ensureDirExists(dirName)
  print("writing team events for game file: %s" % gameFile)
  gameName = os.path.basename(gameFile)
  fileName = os.path.join(dirName, gameName)
  writeEventsToFile(events, fileName)

def readEvents(game, s = ''):
  """return the events of a game, specified as a game id, an absolute
  path to its file, or a relative path to its file"""
  name = filenameForGame(game, s)
  if 'DS_S' not in name:
    print("readEvents: filename = " + name)
    arr = np.loadtxt(name, delimiter=',', dtype='string')
    # arr = np.core.defchararray.replace(arr, 'None', '0')
    events = createEventsFromTxtRows(arr)
  return events

def readSplitEvents(game, s = ''):
  """return the events of a game in a dictionary, specified as a game id, an absolute
  path to its file, or a relative path to its file"""
  name = filenameForGame(game)
  if 'DS_S' not in name:
    print("readSplitEvents: filename = " + name)
    arr = np.loadtxt(name, delimiter = ',', dtype='string')
    events = createSplitEventsFromTxtRows(arr)
  return events

def readSplitValues(game):
  """return the comparables of events of a game in a dictionary"""
  name = filenameForGame(game)
  if 'DS_S' not in name:
  # print("readSplitValues: filename = " + name)
    arr = np.loadtxt(name, delimiter = ',', dtype='string')
    values = createSplitValuesFromTxtRows(arr)
  return values

def readFile(f):
  arr = np.loadtxt(f, delimiter=',', dtype = 'string')
  # arr = np.core.defchararray.replace(arr, 'None', '0')
  return arr

def isGameFile(f):
  # print 'checking', f
  try:
    float(f[0])
    return f[-3:] == 'txt'
  except ValueError:
    return False

def loadGameEvents(f, s = ''):
  f = GAMES_DIR + f
  return readEvents(f)
  # a = readFile(f)
  # return createEventsFromTxtRows(a)

def loadSeasonEvents(s):
  files = listSeasonFiles(s)
  events = []
  for f in files:
    # print 'loading', f
    if isGameFile(f):
      events += readEvents(f, s+'/')
  return events

def loadSeasonSplitValues(f, s):
  f = GAMES_DIR + s + '/' + f

  return readSplitValues(f)

def listGameFiles():
  return os.listdir(GAMES_DIR)

def listSeasonFiles(season):
  return os.listdir(GAMES_DIR + season + '/')

def listTeamFiles(season, team):
  return os.listdir(GAMES_DIR + season + '/' + str(team) + '/')

def getGamesAsEventSeqs():
  files = listGameFiles()
  for f in files:
    yield loadGameEvents(f)

def getSeasonAsSplitValues(s):
  values = {}
  for e in eventNames:
    values[e] = np.array([])

  files = listSeasonFiles(s)
  for f in files:
    if 'DS' not in f:
      e = loadSeasonSplitValues(f, s)
      for name in eventNames:
        if values[name].size == 0:
          values[name] = np.array(e[name])
        else:
          if np.array(e[name]).size != 0:
            values[name] = np.concatenate((values[name], np.array(e[name])))

  return values

def getGameAsSplitValues(s, gameID):
  values = {}
  for e in eventNames:
    values[e] = np.array([])

  e = loadSeasonSplitValues(gameID + '.txt', s)
  for name in eventNames:
    if values[name].size == 0:
      values[name] = np.array(e[name])
    else:
      if np.array(e[name]).size != 0:
        values[name] = np.concatenate((values[name], np.array(e[name])))

  return values

def writeComparables():
  for s in thesisSeasons:
    print "Getting Values %s" % (s)
    values = getSeasonAsSplitValues(s)
    for i in range(len(eventNames)):
      v = values[eventNames[i]]
      f = GAMES_DIR + s + '/' + fileNames[i] + '.txt'
      print "Writing for season %s, event %s, file %s" % (s, eventNames[i], f)
      np.savetxt(f, np.array(v), fmt='%.8f', delimiter=',')

def writeComparablesFiles(values, directory):
  for i in range(len(eventNames)):
    v = values[eventNames[i]]
    f = directory + fileNames[i] + '.txt'
    print "Writing for directory %s, event %s, file %s" % (directory, eventNames[i], f)
    np.savetxt(f, np.array(v), fmt='%.8f', delimiter=',')

def createComparablesDirectories():
  for s in seasons:
    for n in folderNames:
      d = GAMES_DIR + s + '/' + n
      try:
        os.mkdir(d)
      except:
        pass

def readSeasonEventComparables(s, eventFile):
  name = GAMES_DIR + s + '/' + eventFile + '.txt'
  arr = np.loadtxt(name, delimiter=',')
  return arr

def readTeamEventSplitValues(s, teamID, event):
  directory = getTeamDir(s, teamID)
  name = directory + event + '.txt'
  arr = np.loadtxt(name, delimiter=',')
  return arr




############################## Splitting into Teams #####################################
def splitEventsByTeam(events):
  """return a dict of team -> ordered list of events they did"""
  teamDict = {}
  for e in events:
    if e.team not in teamDict:
      print 'Adding team', str(e.team)
      teamDict[e.team] = [e]
    else:
      teamDict[e.team].append(e)
  
  return teamDict

def writeAllTeamEvents():
  files = listGameFiles()
  for f in files:
    events = loadGameEvents(f)
    team2events = splitEventsByTeam(events)
    for team in team2events.keys():
      teamEvents = team2events[team]
      writeTeamEventsForGameFile(team, teamEvents, f)

def listTeamDirs():
  return listSubdirs(TEAMS_DIR)

def listGameFilesForTeam(team):
  teamDir = dirNameForTeam(team)
  return listVisibleFilesInDir(teamDir, endswith='.txt', absPaths=True)

def getGamesForTeam(team):
  gameFiles = listGameFilesForTeam(team)
  # print "team has %d game files" % len(gameFiles)
  for f in gameFiles:
    print("reading games from file " + f)
    yield readEvents(f)

def getTeams():
  return listTeamDirs()

def getSeasonTeamIDs(s):
  d = GAMES_DIR + s + '/'
  return [f for f in os.listdir(d) if os.path.isdir(os.path.abspath(d + f))]

def writeAllTeamComparables(s):
  print 'Getting events'
  # Get all the events of a sesons in array
  events = loadSeasonEvents(s)

  # Get a dictionary that has key-value of team - season events
  teamEvents = splitEventsByTeam(events)
  for team in teamEvents.keys():
    'Writing for team: %s', (str(team))
    # Write original events files
    events = teamEvents[team]
    d = makeTeamDir(team, s)
    writeEventsToFile(events, d + 'events.txt')

    # Write comparables files
    splitValues = createSplitValuesFromEvents(events)
    writeComparablesFiles(splitValues, d)

############################## Document Creation ##################################

# for classification, need to be able to write:
  # for alphabet in Alphabet.getAllAlphabets()
    # for team in Data.getAllTeams()
      # for game in Data.getGamesForTeam(team, alphabet)

############################## Event Exploration ########################################
def eventAfterUnsuccesfulPass(events):
  for i in range(len(events)):
    e = events[i]
    if e.getType() == "PASS":
      if not e.outcome:
        print "Previous Event: %s, team: %d" % (events[i-1].getType(), events[i-1].team)
        print "Next Event: %s, team: %d" % (events[i+1].getType(), events[i+1].team)

############################## Event Creation ###########################################

def createEvents(events, Event):
  return [Event(*e) for e in events]

def createOutcomeEvents(events, Event):
  return [Event(*e[:8], outcome = e[-1]) for e in events]

def createAttackingMoveEvents(events):
  return [AttackingMove(*e[:8], end_x = e[8], end_y = e[9], t = e[10]) for e in events]

def createPassEvents(events):
  return [Pass(*e[:8], end_x = e[8], end_y = e[9], angle = e[10], receiver = e[11], crosses = e[12], header = e[13], free_kick = e[14], corner = e[15], outcome = e[16]) for e in events]

def createShotEvents(events):
  return [Shot(*e[:8], goal_y = e[8], goal_z = e[9], free_kick = max(e[10], e[11]), header = e[12], other_body_part = e[13], on_target = e[14], goal = e[15], penalty = e[16]) for e in events]
  # return [Shot(*e[:8], goal_y = e[8], goal_z = e[9], free_kick = max(e[10], e[11]), header = e[12], other_body_part = e[13], on_target = e[14], goal = e[15]) for e in events]

def createEventFromRow(r):
  t = r[0]
  if t == "PASS":
    e = createPassEventFromRow(r)

  elif t == "SHOT":
    e = createShotEventFromRow(r)

  elif t == "ATTACKING MOVE":
    e = createAttackingMoveEventFromRow(r)

  elif t in frozenset(["AERIAL", "CLEARANCE", "FOUL", "TACKLE", "TAKE ON"]):
    e = createOutcomeEventFromRow(r, eventDict[t])

  else:
    e = eventDict[t](*r[1:9])
  
  return e

def createOutcomeEventFromRow(e, Event):
  return Event(*e[1:9], outcome = e[9])

def createAttackingMoveEventFromRow(e):
  if e[9] == "None":
    e[9] = 0
  if e[10] == "None":
    e[10] = 0
  return AttackingMove(*e[1:9], end_x = e[9], end_y = e[10], t = e[11])

def createPassEventFromRow(e):
  if e[14] == "None":
    e[14] = 0
  if e[12] == 'None':
    e[12] = 0
  return Pass(*e[1:9], outcome = e[9], end_x = e[10], end_y = e[11], angle = e[12], distance = e[13], receiver = e[14], crosses = e[15], header = e[16], free_kick = e[17], corner = e[18])

def createShotEventFromRow(e):
  if e[11] == 'None':
    e[11] = 0
  if e[12] == 'None':
    e[12] = 0
  return Shot(*e[1:9], on_target = e[9], goal = e[10], goal_y = e[11], goal_z = e[12], distance = e[13], free_kick = e[14], header = e[15], other_body_part = e[16], penalty = e[17])

def createEventsFromTxtRows(rows):
  events = []
  for r in rows:
    e = createEventFromRow(r)
    events.append(e)

  return events

def createSplitEventsFromTxtRows(rows):
  events = {}
  for e in eventNames:
    events[e] = []

  for r in rows:
    t = r[0]
    e = createEventFromRow(r)
    events[t].append(e)

  return events

def createSplitValuesFromTxtRows(rows):
  values = {}
  for e in eventNames:
    values[e] = []

  # try:
  for r in rows:
    t = r[0]
    e = createEventFromRow(r)
    values[t].append(e.comparables)
  # except:
  #   print "Error"
  #   print rows

  return values

def createSplitValuesFromEvents(events):
  values = {}
  for e in eventNames:
    values[e] = []

  for event in events:
    values[event.getType()].append(event.comparables)

  return values




############################### List Functions ###########################################


# events - dictionary of arrays, each array refers to a different type of Event
# returns dictionary, each key is a game, each value is an array of all the events for the game
def splitEvents(games, events):
  gameEvents = {}
  splits = [0 for i in range(17)]

  for g in games:
    gameEvents[g] = []

  for n in eventNames:
    for e in events[n]:
      gameEvents[e.game].append(e)

  return gameEvents

def sortEvents(events):
  events.sort()
  return events

def findGameSplit(beg, l, g):
  for i in range(len(l[beg:])):
    if l[beg + i].game != g:
      end = beg + i + 1
      return l[beg:end], end
  return l[beg:], len(l)


############################### Queries ##################################################

def write_games_teams():
  for s in seasons:
    print s
    games = gameQuery(seasonNames[s])
    teams = teamQuery(seasonNames[s])
    g_n = []; t_n = [];
    for g in games:
      g_id = str(g[0])
      g_home = unicodedata.normalize('NFKD', g[1]).encode('ascii', 'ignore')
      g_away = unicodedata.normalize('NFKD', g[2]).encode('ascii', 'ignore')
      g_n.append([g_id, g_home, g_away])
    for t in teams:
      t_id = str(t[0])
      t_name = unicodedata.normalize('NFKD', t[1]).encode('ascii', 'ignore')
      t_n.append([t_id, t_name])
    np.savetxt(s + '-games.txt', np.array(g_n), fmt='%s', delimiter=',')
    np.savetxt(s + '-teams.txt', np.array(t_n), fmt='%s', delimiter=',')

def write_players():
  for s in seasons:
    print s
    players = playerQuery(seasonNames[s])
    p_n = []
    for p in players:
      p_id = str(p[0])
      if p[1] is not None:
        p_first_name = unicodedata.normalize('NFKD', p[1]).encode('ascii', 'ignore')
      else:
        p_first_name = 'Unknown'
      if p[2] is not None:
        p_last_name = unicodedata.normalize('NFKD', p[2]).encode('ascii', 'ignore')
      else:
        p_last_name = 'Unknown'
      p_team = str(p[3])
      p_n.append([p_id, p_first_name, p_last_name, p_team])
    np.savetxt(s + '-players.txt', np.array(p_n), fmt='%s', delimiter=',')

def playerQuery(s):
  query = (s.Player.select(s.Player.id, s.Player.first_name, s.Player.last_name, s.Player.team).tuples())
  return [q for q in query]

def gameQuery(s):
  query = (s.Game.select(s.Game.id, s.Game.home_team_name, s.Game.away_team_name).tuples())
  return [q for q in query]

def teamQuery(s):
  query = (s.Team.select(s.Team.id, s.Team.name).tuples())
  return [q for q in query]

def eventQuery(s, model):
  query = (model
            .select(model.game,
                    model.period,
                    model.min,
                    model.sec,
                    model.player,
                    model.team,
                    model.x,
                    model.y)
            .order_by(model.game,
                      model.period,
                      model.min,
                      model.sec)
            .tuples())

  return [q for q in query]

def eventOutcomeQuery(s, model):
  query = (model
            .select(model.game,
                    model.period,
                    model.min,
                    model.sec,
                    model.player,
                    model.team,
                    model.x,
                    model.y,
                    model.outcome)
            .order_by(model.game,
                      model.period,
                      model.min,
                      model.sec)
            .tuples())

  return [q for q in query]

def attackingMoveQuery(s):
  model = s.AttackingMove
  query = (model
            .select(model.game,
                    model.period,
                    model.min,
                    model.sec,
                    model.player,
                    model.team,
                    model.x,
                    model.y,
                    model.end_x,
                    model.end_y,
                    model.type)
            .order_by(model.game,
                      model.period,
                      model.min,
                      model.sec)
            .tuples())

  return [q for q in query]

def passQuery(s):
  model = s.Pass
  query = (model
            .select(model.game,
                    model.period,
                    model.min,
                    model.sec,
                    model.player_id_1,
                    model.team,
                    model.start_x,
                    model.start_y,
                    model.end_x,
                    model.end_y,
                    model.angle,
                    model.player_id_2,
                    model.crosses,
                    model.header,
                    model.free_kick,
                    model.corner,
                    model.outcome)
            .order_by(model.game,
                      model.period,
                      model.min,
                      model.sec)
            .tuples())

  return [q for q in query]

def shotQuery(s):
  model = s.Shot
  query = (model
            .select(model.game,
                    model.period,
                    model.min,
                    model.sec,
                    model.player,
                    model.team,
                    model.start_x,
                    model.start_y,
                    model.goal_y,
                    model.goal_z,
                    model.free_kick,
                    model.direct_free_kick,
                    model.header,
                    model.other_body_part,
                    model.on_target,
                    model.goal,
                    model.penalty)
            .order_by(model.game,
                      model.period,
                      model.min,
                      model.sec)
            .tuples())

  return [q for q in query]

def aerialQuery(s):
  model = s.Aerial
  return eventOutcomeQuery(s, model)

def ballRecoveryQuery(s):
  model = s.BallRecovery
  return eventQuery(s, model)

def blockQuery(s):
  model = s.Block
  return eventQuery(s, model)

def challengeQuery(s):
  model = s.Challenge
  return eventQuery(s, model)

def clearanceQuery(s):
  model = s.Clearance
  return eventOutcomeQuery(s, model)

def dispossessedQuery(s):
  model = s.Dispossessed
  return eventQuery(s, model)

def errorQuery(s):
  model = s.Error
  return eventQuery(s, model)

def foulQuery(s):
  model = s.Foul
  return eventOutcomeQuery(s, model)

def interceptionQuery(s):
  model = s.Interception
  return eventQuery(s, model)

def keeperQuery(s):
  model = s.KeeperEvent
  return eventQuery(s, model)

def tackleQuery(s):
  model = s.Tackle
  return eventOutcomeQuery(s, model)

def takeOnQuery(s):
  model = s.TakeOn
  return eventOutcomeQuery(s, model)

def touchQuery(s):
  model = s.Touch
  return eventQuery(s, model)

def turnoverQuery(s):
  model = s.Turnover
  return eventQuery(s, model)
