import numpy as np
import math

longest = 19

class EventDictionary:

  def dictionary():
    return {"ATTACKING MOVE": AttackingMove, "PASS": Pass, "SHOT": Shot, "AERIAL": Aerial, "CLEARANCE": Clearance, "FOUL": Foul, "TACKLE": Tackle, "TAKE ON": TakeOn, "BALL RECOVERY": BallRecovery, "BLOCK": Block,
              "CHALLENGE": Challenge, "DISPOSSESSED": Dispossessed, "ERROR": Error, "INTERCEPTION": Interception, "KEEPER EVENT": KeeperEvent, "TOUCH": Touch, "TURNOVER": Turnover}

class EventTypes:
  ATTACKING_MOVE = "ATTACKING MOVE"
  PASS = "PASS"
  SHOT = "SHOT"
  AERIAL = "AERIAL"
  CLEARANCE = "CLEARANCE"
  FOUL = "FOUL"
  TACKLE = "TACKLE"
  TAKE_ON = "TAKE ON"
  BALL_RECOVERY = "BALL RECOVERY"
  BLOCK = "BLOCK"
  CHALLENGE = "CHALLENGE"
  DISPOSSESSED = "DISPOSSESSED"
  ERROR = "ERROR"
  INTERCEPTION = "INTERCEPTION"
  KEEPER_EVENT = "KEEPER EVENT"
  TOUCH = "TOUCH"
  TURNOVER = "TURNOVER"

class Event:

  def __init__(self, game, period, minute, sec, player, team, start_x, start_y):
    self.game = int(game)
    self.start_x = float(start_x)
    self.start_y = float(start_y)
    self.min = float(minute)
    self.sec = float(sec)
    self.period = float(period)
    self.player = int(player)
    self.team = int(team)
    self.comparables = [self.start_x, self.start_y]

  def toRow(self):
    l = [self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def __eq__(self, other):
    # loc_x = self.start_x == other.start_x
    # loc_y = self.start_y == other.start_y

    time_game = self.game == other.game
    time_min = self.min == other.min
    time_sec = self.sec == other.sec
    time_period = self.period == other.period

    # player = self.player == other.player
    # team = self.team == other.team

    return (time_game and time_min and time_sec and time_period)

  def __ne__(self,other):
    return not(self == other)

  def __lt__(self, other):
    if self.game == other.game:
      if self.period == other.period:
        if self.min == other.min:
          if self.sec == other.sec:
            return False
          else:
            return self.sec < other.sec
        else:
          return self.min < other.min
      else:
        return self.period < other.period
    else:
      return self.game < other.game

  def __gt__(self, other):
    if self.game == other.game:
      if self.period == other.period:
        if self.min == other.min:
          if self.sec == other.sec:
            return False
          else:
            return self.sec > other.sec
        else:
          return self.min > other.min
      else:
        return self.period > other.period
    else:
      return self.game > other.game

  def __le__(self, other):
    if self.game == other.game:
      if self.period == other.period:
        if self.min == other.min:
          if self.sec == other.sec:
            return True
          else:
            return self.sec < other.sec
        else:
          return self.min < other.min
      else:
        return self.period < other.period
    else:
      return self.game > other.game

  def __ge__(self, other):
    if self.game == other.game:
      if self.period == other.period:
        if self.min == other.min:
          if self.sec == other.sec:
            return True
          else:
            return self.sec > other.sec
        else:
          return self.min > other.min
      else:
        return self.period > other.period
    else:
      return self.game > other.game


class Aerial(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.outcome = int(kwargs.get('outcome'))
    self.comparables = [self.start_x, self.start_y, self.outcome]

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.AERIAL

  def __str__(self):
    return str(self.getType())

class AttackingMove(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    if kwargs.get('end_x') is None:
      self.end_x = 0.0
    else:
      self.end_x = float(kwargs.get('end_x'))
    
    if kwargs.get('end_y') is None:
      self.end_y = 0.0
    else:
      self.end_y = float(kwargs.get('end_y'))

    self.t = kwargs.get('t')


  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.end_x, self.end_y, self.t]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.ATTACKING_MOVE

  def __str__(self):
    return str(self.getType())

class BallRecovery(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.BALL_RECOVERY

  def __str__(self):
    return str(self.getType())

class Block(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.BLOCK

  def __str__(self):
    return str(self.getType())

class Challenge(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.CHALLENGE

  def __str__(self):
    return str(self.getType())

class Clearance(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.outcome = int(kwargs.get('outcome'))
    self.comparables = [self.start_x, self.start_y, self.outcome]

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.CLEARANCE

  def __str__(self):
    return str(self.getType())

class Dispossessed(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.DISPOSSESSED

  def __str__(self):
    return str(self.getType())

class Error(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.ERROR

  def __str__(self):
    return str(self.getType())

class Foul(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.outcome = int(kwargs.get('outcome'))
    self.comparables = [self.start_x, self.start_y, self.outcome]

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.FOUL

  def __str__(self):
    return str(self.getType())

class Interception(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.INTERCEPTION

  def __str__(self):
    return str(self.getType())

class KeeperEvent(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.KEEPER_EVENT

  def __str__(self):
    return str(self.getType())

class Pass(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.end_x = float(kwargs.get('end_x'))
    self.end_y = float(kwargs.get('end_y'))
    self.angle = Pass.calcAnglePass(self.end_x, self.end_y, self.start_x, self.start_y)
    self.distance = self.calcDistancePass()

    if kwargs.get('receiver') is None:
      self.receiver = 0
    else:
      self.receiver = int(kwargs.get('receiver'))

    self.crosses = int(kwargs.get('crosses'))
    self.header = int(kwargs.get('header'))
    self.free_kick = int(kwargs.get('free_kick'))
    self.corner = int(kwargs.get('corner'))

    self.outcome = int(kwargs.get('outcome'))

    self.comparables = [self.start_x, self.start_y, self.outcome, self.end_x, self.end_y, self.angle, self.distance, self.crosses, self.header, self.free_kick, self.corner]

  def calcDistancePass(self):
    distance = (((float(self.end_x) - float(self.start_x)) ** 2) + ((float(self.end_y) - float(self.start_y)) ** 2)) ** 0.5
    return distance


  @staticmethod
  def calcAnglePass(end_x, end_y, start_x, start_y):
    start = np.array([1.0, 0.0])
    end = np.array([end_x - start_x, end_y - start_y])

    start = start / np.linalg.norm(start)
    if np.linalg.norm(end) == 0:
      end = end + np.array([0.0001, 0.0])
    end = end / np.linalg.norm(end)

    angle = np.arccos(np.dot(start, end))
    if np.isnan(angle):
      if (start == end).all():
        return 0.0
      else:
        return np.pi
    if end_y < start_y:
      return 2 * np.pi - angle
    return angle

  def toRow(self):
    return [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome, self.end_x, self.end_y, self.angle, self.distance, self.receiver, self.crosses, 
            self.header, self.free_kick, self.corner]

  def getType(self):
    return EventTypes.PASS

  def __str__(self):
    return str(self.getType())

class Shot(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)

    if kwargs.get('goal_y') is None:
      self.goal_y = 0.0
    else:
      self.goal_y = float(kwargs.get('goal_y'))
    
    if kwargs.get('goal_z') is None:
      self.goal_z = 0.0
    else:
      self.goal_z = float(kwargs.get('goal_z'))

    self.distance = self.calcDistanceShot()
    self.free_kick = float(kwargs.get('free_kick'))
    self.penalty = float(kwargs.get('penalty', 0))

    self.header = float(kwargs.get('header'))
    self.other_body_part = float(kwargs.get('other_body_part'))

    self.on_target = float(kwargs.get('on_target'))
    self.goal = float(kwargs.get('goal'))

    self.comparables = [self.start_x, self.start_y, self.goal_y, self.goal_z, self.distance, self.free_kick, self.penalty, self.header, self.other_body_part, self.on_target, self.goal]

  def calcDistanceShot(self):
    distance = (((100.0 - float(self.start_x)) ** 2) + ((50.0 - float(self.start_y)) ** 2)) ** 0.5
    return distance

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.on_target, self.goal, self.goal_y, self.goal_z, self.distance, self.free_kick, self.header,
          self.other_body_part, self.penalty]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.SHOT

  def __str__(self):
    return str(self.getType())

class Tackle(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.outcome = int(kwargs.get('outcome'))
    self.comparables = [self.start_x, self.start_y, self.outcome]

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.TACKLE

  def __str__(self):
    return str(self.getType())


class TakeOn(Event):

  def __init__(self, *args, **kwargs):

    Event.__init__(self, *args)
    self.outcome = int(kwargs.get('outcome'))
    self.comparables = [self.start_x, self.start_y, self.outcome]

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y, self.outcome]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.TAKE_ON

  def __str__(self):
    return str(self.getType())


class Touch(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.TOUCH

  def __str__(self):
    return str(self.getType())

class Turnover(Event):

  def toRow(self):
    l = [self.getType(), self.game, self.period, self.min, self.sec, self.player, self.team, self.start_x, self.start_y]
    s = longest - len(l)
    z = [0 for i in range(s)]
    return l + z

  def getType(self):
    return EventTypes.Turnover

  def __str__(self):
    return str(self.getType())
