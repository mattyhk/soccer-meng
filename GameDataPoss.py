from BallEventsPoss import *
from peewee import *
# from utils import *

import os
import numpy as np


GAMES_DIR = 'Files/Possession/'
TEAMS_DIR = 'Files/Possession/'
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

def getAllTeamDir(s):
  return [os.path.join('Files/Possession/' + s,n) for n in os.listdir('Files/Possession/' + s) if os.path.isdir(os.path.join('Files/Possession/' + s,n))]

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

def readPossessionEventsValues(game, s = ''):
  """return the events of a game, specified as a game id, an absolute
  path to its file, or a relative path to its file"""
  name = filenameForGame(game, s)
  if 'DS_S' not in name:
    print("readPossessionEvents: filename = " + name)
    arr = np.loadtxt(name, delimiter=',', dtype='string')
    # arr = np.core.defchararray.replace(arr, 'None', '0')
    values, events = createPossessionValuesFromTxtRows(arr)
  return values, events

def readFile(f):
  arr = np.loadtxt(f, delimiter=',', dtype = 'string')
  # arr = np.core.defchararray.replace(arr, 'None', '0')
  return arr

def loadGameEvents(f, s = ''):
  if os.path.isfile(f):
    return readEvents(f)
  else:
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

def isGameFile(f):
  # print 'checking', f
  try:
    float(f[0])
    return f[-3:] == 'txt'
  except ValueError:
    return False

def checkChronologicalOrder():
  files = listGameFiles()
  for f in files:
    if isGameFile(f):
      game = readFile(GAMES_DIR + f)
      events = createEventsFromTxtRows(game)
      for i in range(len(events)):
        if i + 1 != len(events):
          e_1 = events[i]
          e_2 = events[i + 1]
          if (e_1.period > e_2.period):
            print f
            print e_1.getType(), ' happens at', e_1.period, 'after ', e_2.getType(), 'at', e_2.period
          if (e_1.period == e_2.period) and (e_1.min > e_2.min):
          # if (e_1.min > e_2.min):
            print f
            print e_1.getType(), ' happens at', e_1.min, 'after ', e_2.getType(), 'at', e_2.min
          if (e_1.period == e_2.period) and (e_1.min == e_2.min) and (e_1.sec > e_2.sec):
            print f
            print e_1.getType(), ' happens at', e_1.sec, 'after ', e_2.getType(), 'at', e_2.sec


def createPossessionFiles():
  # For every season
  # Create season directory in Possession folder
  # Load all events of each game in new format
  # Write all events for each game in new format
  # Write all the comparables
  # Split events into the teams
  # For each team create a team directory
  # Write all the events
  # Write all the comparables

  thesisSeasons = ['algue13']
  
  for s in thesisSeasons:

    print 'Getting files for', s

    values = {}
    for e in eventNames:
      values[e] = []

    season_events = []

    seasonDir = 'Files/Thesis/' + s + '/'
    seasonPossDir = GAMES_DIR + s + '/'
    os.mkdir(seasonPossDir)

    files = os.listdir('Files/Thesis/' + s + '/')
    for f in files:
      if isGameFile(f):
        print 'Reading file', f
        arr = np.loadtxt('Files/Thesis/' + s + '/' + f, delimiter=',', dtype='string')
        v, e = createPossessionValuesFromTxtRows(arr)
        for name in eventNames:
          values[name] += v[name]
        season_events += e
        writeEventsToFile(e, GAMES_DIR + s + '/' + f)

    writeComparablesFiles(values, GAMES_DIR + s + '/')

    teamEvents = splitEventsByTeam(season_events)
    for team in teamEvents.keys():
      # 'Writing for team: %s', (str(team))
      # Write original events files
      events = teamEvents[team]
      d = makeTeamDir(team, s)
      writeEventsToFile(events, d + 'events.txt')

      # Write comparables files
      splitValues = createSplitValuesFromEvents(events)
      writeComparablesFiles(splitValues, d)


def count_events_for_teams(s):
  teams = [n for n in os.listdir('Files/Possession/' + s) if os.path.isdir(os.path.join('Files/Possession/' + s,n))]
  counts = {}
  for t in teams:
    counts[t] = 0
    d = 'Files/Possession/' + s + '/' + t
    for f in os.listdir(d):
      print os.path.join(d,f)
      e = readFile(os.path.join(d,f))
      counts[t] += e.shape[0]
  return counts







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
  # Get all the events of a season in array
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

# def createEventFromRow(r, poss_pos, poss_shot, poss_goal, poss_off_entry, poss_pass = None):
def createEventFromRow(r):
  t = r[0]
  if t == "PASS":
    # e = createPassEventFromRow(r, poss_pos, poss_shot, poss_goal, poss_off_entry, poss_pass)
    e = createPassEventFromRow(r)

  elif t == "SHOT":
    # e = createShotEventFromRow(r, poss_pos, poss_shot, poss_goal, poss_off_entry)
    e = createShotEventFromRow(r)

  elif t == "ATTACKING MOVE":
    e = createAttackingMoveEventFromRow(r)

  elif t in frozenset(["AERIAL", "CLEARANCE", "FOUL", "TACKLE", "TAKE ON"]):
    # e = createOutcomeEventFromRow(r, eventDict[t], poss_pos, poss_shot, poss_goal, poss_off_entry)
    e = createOutcomeEventFromRow(r, eventDict[t])

  else:
    # e = eventDict[t](*(np.append(r[1:9], [poss_pos, poss_shot, poss_goal, poss_off_entry])))
    e = eventDict[t](*r[1:13])
  
  return e

# def createOutcomeEventFromRow(e, Event, poss_pos, poss_shot, poss_goal, poss_off_entry):
def createOutcomeEventFromRow(e, Event):
  return Event(*(e[1:13]), outcome = e[13])

def createAttackingMoveEventFromRow(e):
  if e[9] == "None":
    e[9] = 0
  if e[10] == "None":
    e[10] = 0
  return AttackingMove(*e[1:9], end_x = e[9], end_y = e[10], t = e[11])

# def createPassEventFromRow(e, poss_pos, poss_shot, poss_goal, poss_off_entry, poss_pass):
def createPassEventFromRow(e):
  if e[14] == "None":
    e[14] = 0
  if e[12] == 'None':
    e[12] = 0
  return Pass(*(e[1:13]), outcome = e[13], end_x = e[14], end_y = e[15], angle = e[16], distance = e[17], receiver = e[18], crosses = e[19], header = e[20], free_kick = e[21], corner = e[22], possession_pass = e[23])

# def createShotEventFromRow(e, poss_pos, poss_shot, poss_goal, poss_off_entry):
def createShotEventFromRow(e):
  if e[11] == 'None':
    e[11] = 0
  if e[12] == 'None':
    e[12] = 0
  return Shot(*(e[1:13]), on_target = e[13], goal = e[14], goal_y = e[15], goal_z = e[16], distance = e[17], free_kick = e[18], header = e[19], other_body_part = e[20], penalty = e[21])

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

def createPossessionValuesFromTxtRows(rows):
  # print '--------------------------------------------- START --------------------------------------------'
  values = {}
  for e in eventNames:
    values[e] = []

  events_all = []

  begin_possession = 0

  while begin_possession < len(rows) - 1:

    # print "First event of possession is row", begin_possession
    full_events = []
    possession = []
    possession_minor = []
    possession_position = 1
    pass_possession_position = 1
    possession_shot = 0
    possession_goal = 0
    possession_off_entry = 0

    possession_first_row = rows[begin_possession]
    possession_period = float(possession_first_row[2])
    possession_team = int(possession_first_row[6])

    for i in xrange(len(rows[begin_possession:])):

      # Current row
      r = rows[begin_possession + i]
      t = r[0]

      x = float(r[7])
      y = float(r[8])

      if t == 'TOUCH' or t == 'KEEPER EVENT' or t == 'BALL RECOVERY':
        # print 'Passing Touch'
        continue

      # Check if the period is the same
      elif float(r[2]) != possession_period:
        # print 'Changed Period'
        break

      # Check if the team is the same
      elif int(r[6]) != possession_team:
        if t == 'TAKE ON' or t == 'AERIAL' or t == 'TACKLE':
          if float(r[9]) == 0.0:
            possession_minor.append((t, r, 1, None))
            full_events.append((t, r, 1, None))
            # print t
          else:
            # print 'Changed Team'
            break

        elif t == 'FOUL' or t == 'DISPOSSESSED' or t == 'ERROR':
          possession_minor.append((t, r, 1, None))
          full_events.append((t, r, 1, None))
          # print t

        else:
          # print 'Changed Team'
          break

      else:
        # if t == 'TOUCH':
        #   print 'Touch'
        #   possession.append((t, r, possession_position, None))

        if 80 <= x < 100 and 20 < y < 80:
          possession_off_entry = 1
      
        if t == 'PASS':
          # print 'Pass'
          possession.append((t, r, possession_position, pass_possession_position))
          full_events.append((t, r, possession_position, pass_possession_position))
          pass_possession_position += 1
        
        elif t == 'SHOT':
          # print 'Shot'
          possession_shot = 1
          if float(r[10]) == 1.0:
            # print 'Goal'
            possession_goal = 1
          possession.append((t, r, possession_position, None))
          full_events.append((t, r, possession_position, None))

        elif t == 'FOUL' or t == 'ERROR':
          possession.append((t, r, possession_position, None))
          full_events.append((t, r, possession_position, None))
          # print t
          i += 1
          break

        else:
          # print t
          possession.append((t, r, possession_position, None))
          full_events.append((t, r, possession_position, None))
        
        possession_position += 1

    begin_possession += max(i, 1)

    for event_tuple in possession_minor:
      t = event_tuple[0]
      e = createEventFromRow(event_tuple[1], event_tuple[2], possession_shot, possession_goal, possession_off_entry, event_tuple[3])
      values[t].append(e.comparables)
      # print e.team, t, e.comparables

    for event_tuple in possession:
      t = event_tuple[0]
      e = createEventFromRow(event_tuple[1], event_tuple[2], possession_shot, possession_goal, possession_off_entry, event_tuple[3])
      values[t].append(e.comparables)
      # print e.team, t, e.comparables

    for event_tuple in full_events:
      e = createEventFromRow(event_tuple[1], event_tuple[2], possession_shot, possession_goal, possession_off_entry, event_tuple[3])
      events_all.append(e)

  return values, events_all


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

def gameQuery(s):
  query = (s.Game.select(s.Game.id))
  return [q.id for q in query]

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
