import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.cluster import KMeans

# import GameData
# from BallEvents import EventTypes
import GameDataPoss
from BallEventsPoss import EventTypes

FIG_PATH = 'Clustering/Figures/paper/'
PLAYING_AREA = 115.0 * 75.0

class Values:

  index = {'game': 0, 'x': 1, 'y': 2, 'poss_order': 3, 'poss_shot': 4, 'poss_goal': 5, 'poss_off_entry': 6, 
            'outcome': 7, 'angle': 10, 'distance': 11, 'crosses': 12, 'header': 13, 'free_kick': 14, 'corner': 15, 'poss_pass': 16,
              'shot_distance': 9, 'shot_free_kick': 10, 'penalty': 11, 'shot_header': 12, 'body_part': 13, 'on_target': 14, 'goal': 15,
                'end_x': 8, 'end_y': 9}

  def __init__(self, data, shot = False, passes = False):

    self.data = data

    self.x = data[:, Values.index['x']]
    self.y = data[:, Values.index['y']]
    self.shot = shot
    self.passes = passes

    if data.shape[1] > 2:
      self.outcome = data[:, Values.index['outcome']]

    if shot:
      self.shotDistance = data[:, Values.index['shot_distance']]
      self.shot_free_kick = data[:, Values.index['shot_free_kick']]
      self.penalty = data[:, Values.index['penalty']]
      self.header = data[:, Values.index['shot_header']]
      self.body_part = data[:, Values.index['body_part']]
      self.on_target = data[:, Values.index['on_target']]
      self.goal = data[:, Values.index['goal']]

    if passes:
      self.angle = data[:, Values.index['angle']]
      self.distance = data[:, Values.index['distance']]
      self.crosses = data[:, Values.index['crosses']]
      self.header = data[:, Values.index['header']]
      self.free_kick = data[:, Values.index['free_kick']]
      self.corner = data[:, Values.index['corner']]

    self.transformCoordinates()

  def addData(self, data):
    self.data = np.concatenate(self.data, data)

  def getValues(self, value):
    try:
      return self.data[:,  Values.index[value]].copy()
    except:
      print "Does not contain value %s", (value)
      return

  def getSlicedValues(self, values):
    indices = [Values.index[v] for v in values]
    return self.data[:, indices].copy()

  def updateValues(self, value, newValue):
    self.data[:, Values.index[value]] = newValue

  def transformCoordinates(self):
    xValues = self.getValues('x')
    yValues = self.getValues('y')
    xNewValues = coordinateScaling(xValues, 'x')
    yNewValues = coordinateScaling(yValues, 'y')
    self.updateValues('x', xNewValues)
    self.updateValues('y', yNewValues)
    if self.passes:
      xValues = self.getValues('end_x')
      yValues = self.getValues('end_y')
      xNewValues = coordinateScaling(xValues, 'x')
      yNewValues = coordinateScaling(yValues, 'y')
      self.updateValues('end_x', xNewValues)
      self.updateValues('end_y', yNewValues)
      self.updateValues('distance', scaleDistances(self.getValues('x'), self.getValues('y'), self.getValues('end_x'), self.getValues('end_y')))



##################################### Set Up ####################################################################
def getSeasonsValues(event, seasons):
  if not isinstance(seasons, basestring):
    first = True
    v = []
    for s in seasons:
      if first:
        v = GameDataPoss.readSeasonEventComparables(s, event)
        first = False
      else:
        v = np.concatenate(v, GameDataPoss.readSeasonEventComparables(s, event))
  else:
    v = GameDataPoss.readSeasonEventComparables(seasons, event)
  return Values(v, passes = event == 'pass', shot = event == 'shot')

def getGameValues(event, game, season):
  v = GameDataPoss.getGameAsSplitValues(season, game)
  return Values(v[event], passes = event == 'pass', shot = event == 'shot')

def getTeamValues(event, team, season):
  events = GameDataPoss.readTeamEventSplitValues(season, team, event)
  return Values(events, passes = event == 'pass', shot = event == 'shot')

def getAllTeamValues(event, seasons):
  team_values = {}
  for season in seasons:
    teams = GameDataPoss.getSeasonTeamIDs(season)
    for t in teams:
      v = getTeamValues(event, t, season)
      team_values[t] = v
  return team_values
  



##################################### Transforming Events #######################################################

#### Transform Coordinates ####
def coordinateScaling(values, c):
  if c == 'x':
    newVal = values * 115.0 / 100.0
  elif c == 'y':
    newVal = values * 75.0 / 100.0
  else:
    raise ValueError(('Not an allowed value %s', (c)))
  return newVal

##### Rescale Distances ######
def scaleDistances(start_x, start_y, end_x, end_y):
  d = (((end_x - start_x) ** 2) + ((end_y - start_y) ** 2)) ** 0.5
  return d 


####### Rescale Angles #######
########### To Do ############
def scaleAngles(start_x, start_y, end_x, end_y):
  start = np.array([[1.0, 0.0]])
  start = np.repeat(start, start_x.size, axis=0)

#### Percentile Scaling #####
def percentileScaling(values, maxPercentile, minPercentile):
  maxPercentileValue = np.percentile(values, maxPercentile, axis = 0)
  print maxPercentileValue
  minPercentileValue = np.percentile(values, minPercentile, axis = 0)
  print minPercentileValue
  normalizedValues = (values - minPercentileValue) / (maxPercentileValue - minPercentileValue)
  normalizedValues = np.nan_to_num(normalizedValues)
  return normalizedValues

#### Z-Score Normalization ####
def zScoreNormalization(values):
  mean = np.mean(values, axis=0)
  stdVal = np.std(values, axis=0)
  normalizedValues = (values - mean) / (stdVal)
  normalizedValues = np.nan_to_num(normalizedValues)
  # normalizedValues = StandardScaler().fit_transform(values)
  return normalizedValues

def scaleIndices(scaling, indices, data):
  """ Only scale the indices listed """
  for i in indices:
    a = data[:, i].copy()
    a = scaling(a)
    data[:, i] = a
  return data



##################################### Distance Metric ############################################################

INFINITY = 1e49
def manhattanDistance(eventA, eventB):
  if eventA.getType() != eventB.getType():
    return INFINITY
  else:
    diff = np.array(eventA.comparables) - np.array(eventB.comparables)
    return sum(diff)

##################################### 1-D, 2-D Exploration Plotting ########################################################
def get_bins(NUM_BINS = 20):
  # x = [115.0 / NUM_BINS * i for i in xrange(NUM_BINS + 1)]
  # y = [75.0 / NUM_BINS * i for i in xrange(NUM_BINS + 1)]
  x = np.linspace(0,115.0, 7)
  y = np.linspace(0,75.0,4)
  return [y, x]

def histogram(data, numBins = None, weights = None, title = "", xlabel = "", ylabel = "Counts"):
  if numBins is not None:
    counts, edges = np.histogram(data, bins = numBins, density = True)
  else:
    counts, edges = np.histogram(data, density = True)
  plt.figure()
  plt.hist(data, edges, color = 'r', weights = weights)
  plt.suptitle(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def hexHeatmap(data, numBins = 100, title = "", xlabel = "", ylabel = ""):
  x = data[0]
  y = data[1]
  plt.figure()
  plt.hexbin(x, y, cmap = "Reds", gridsize = numBins)
  plt.suptitle(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.colorbar()
  plt.show()

def histo2d(data, numBins = get_bins(), weights = None, ranged = False, title = "", xlabel = "x", ylabel = "y", save = False, filename = ''):
  if ranged:
    counts, yedges, xedges = np.histogram2d(data[1], data[0], bins = numBins, weights = weights, range = [[0, 100], [0, 100]], normed = True)
    height = yedges[1] - yedges[0]
    width = xedges[1] - xedges[0]
    counts = counts * height * width
  else:
    counts, yedges, xedges = np.histogram2d(data[1], data[0], bins = numBins, weights = weights, normed = True)
    height = yedges[1] - yedges[0]
    width = xedges[1] - xedges[0]
    counts = counts * height * width

  extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

  plt.figure()

  # plt.imshow(counts, extent = extent, cmap = 'Reds', interpolation = 'nearest')
  plt.imshow(counts, extent = extent, cmap = 'Reds')
  plt.suptitle(title)
  # plt.suptitle("2012-13 La Liga Pass Frequencies")
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  cbar = plt.colorbar()
  cbar.ax.set_ylabel('Frequencies')

  # # Annotate diagram
  x_centre = coordinateScaling(50.0, 'x')
  y_centre_start = 0.0
  y_centre_end = 74.8
  x_defensive_box_end = coordinateScaling(17.0, 'x')
  x_defensive_box_start = 0.2
  x_attack_box_end = 114.8
  x_attack_box_start = coordinateScaling(83.0, 'x')
  y_box_bottom = coordinateScaling(21.1, 'y')
  y_box_top = coordinateScaling(78.9, 'y')

  plt.plot([x_centre, x_centre], [y_centre_start, y_centre_end], 'k-')
  plt.plot([x_defensive_box_start, x_defensive_box_end], [y_box_top, y_box_top], 'k-')
  plt.plot([x_defensive_box_start, x_defensive_box_end], [y_box_bottom, y_box_bottom], 'k-')
  plt.plot([x_attack_box_start, x_attack_box_end], [y_box_top, y_box_top], 'k-')
  plt.plot([x_attack_box_start, x_attack_box_end], [y_box_bottom, y_box_bottom], 'k-')
  plt.plot([x_defensive_box_end, x_defensive_box_end], [y_box_bottom, y_box_top], 'k-')
  plt.plot([x_attack_box_start, x_attack_box_start], [y_box_bottom, y_box_top], 'k-')

  plt.xlim([0.0, 115.0])
  plt.ylim([0.0, 75.0])

  for x_tick in numBins[1][1:-1]:
    plt.axvline(x=x_tick, linewidth=0.5, linestyle='--', color='k')
  for y_tick in numBins[0][1:-1]:
    plt.axhline(y=y_tick, linewidth=0.5, linestyle='--', color='k')


  if save:
    plt.savefig(FIG_PATH + filename, format='eps', dpi=800)
    return

  else:
    plt.show()

def multiHisto(data_array, label_array, numBins = 10, weights = None, title = "", xlabel = "", ylabel = 'Counts', save = False, filename = ''):
  plt.figure()
  plt.hist(data_array, label = label_array)
  plt.suptitle(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend(loc='upper right')

  if save:
    plt.savefig(FIG_PATH + filename)
    return

  else:
    plt.show()



##################################### PCA #########################################################################

def pcaDecomp(data, normalize = True):
  if normalize:
    data = StandardScaler().fit_transform(data)

  pca = sklearnPCA(n_components = 2)
  decomp = pca.fit_transform(data)
  # plt.scatter(data[:,0], data[:,1])
  # plt.show()
  histo2d(decomp, ranged = False)

################################################# Classification ###################################################
def calc_accuracy(X_predict, y_true):
  score = np.array(X_predict == y_true)
  score = score.astype(int)
  accuracy = float(sum(score)) / float(score.shape[0])
  return accuracy

def calc_euclidean_distance(array_1, array_2):
  return np.linalg.norm(array_1 - array_2)

def create_distance_matrix(X):
  condensed_distances = pdist(X)
  distance_matrix = squareform(condensed_distances)
  return distance_matrix

def create_test_split(X, y, test_size = 0.3):
  # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = test_size)
  sss = cross_validation.StratifiedShuffleSplit(y, 1, test_size = test_size)
  for train, test in sss:
    train_indices = train
    test_indices = test
    # print train, test
  X_train = X[(train_indices)]
  y_train = y[(train_indices)]
  X_test = X[(test_indices)]
  y_test = y[(test_indices)]
  return X_train, X_test, y_train, y_test

# Splits season into num_splits random sets
def random_split(data, num_splits = 4):
  length = data.shape[0]
  random_array = np.random.rand(length)
  split_data = []
  for i in range(num_splits):
    lb = i / float(num_splits)
    ub = (i + 1.0) / float(num_splits)
    split_data.append(data[(random_array >= lb) & (random_array < ub)])
  return split_data, num_splits

# Need to Fix
# def random_sampling(data, num_samples = 100):
#   length = data.shape[0]
#   sample_data = []
#   UB = 0.25
#   for i in range(num_samples):
#     random_array = np.random.rand(length)
#     sample = data[(random_array < UB)]
#     sample_data.append(sample)
#   return np.array(sample_data), num_samples

# Splits season contiguously
def split_season(data, num_splits = 2):
  length = data.shape[0]
  split_data = []
  for i in xrange(num_splits):
    split_data.append(data[i * length / num_splits : (i + 1) * length / num_splits])
  return np.array(split_data), num_splits

# Splits season based on games
def split_games(team_data):
  data = team_data.getSlicedValues(['x', 'y'])
  game_ids = np.array(team_data.getValues('game'), dtype=int)
  unique_game_ids = np.unique(game_ids)

  num_games = len(unique_game_ids)
  split_data = []
  
  for i in unique_game_ids:
    split_data.append(data[(game_ids == i)])

  return np.array(split_data), num_games

def get_average_distances_between_teams(d_matrix, num_teams, num_samples_per_team):
  averages = np.zeros((num_teams, num_teams))

  for i in xrange(num_teams):
    for j in xrange(i + 1):
      row_min = i * num_samples_per_team
      row_max = (i + 1) * num_samples_per_team
      col_min = j * num_samples_per_team
      col_max = (j + 1) * num_samples_per_team
      d = d_matrix[row_min:row_max, col_min:col_max]
      
      if i == j:
        d = d[np.tril_indices(d.shape[0], k=-1)]

      averages[i][j], averages[j][i] = np.mean(d), np.mean(d)

  return averages

def get_max_distances_between_teams(d_matrix, num_teams, num_samples_per_team):
  maxes = np.zeros((num_teams, num_teams))

  for i in xrange(num_teams):
    for j in xrange(i + 1):
      row_min = i * num_samples_per_team
      row_max = (i + 1) * num_samples_per_team
      col_min = j * num_samples_per_team
      col_max = (j + 1) * num_samples_per_team
      d = d_matrix[row_min:row_max, col_min:col_max]

      if i == j:
        d = d[np.tril_indices(d.shape[0], k=-1)]

      maxes[i][j], maxes[j][i] = np.amax(d), np.amax(d)

  return maxes

def get_histo2d(values, numBins = get_bins()):
  counts, yedges, xedges = np.histogram2d(values.getValues('y'), values.getValues('x'), bins = numBins, normed = True)
  height = yedges[1] - yedges[0]
  width = xedges[1] - xedges[0]
  counts = counts * height * width
  return counts

def get_histo(values, numBins = 10, weights = None):
  counts, edges = np.histogram(values, bins = numBins, weights = weights, density = True)
  return counts

def get_custom_histo2d(data):
  return

def get_team_histo2d_features(data, teams):
  y = []
  x = []

  # For each team, sample data
  # Calculate normalised histogram for each part
  for t in teams:
    team_values = data[t].getSlicedValues(['x', 'y'])
    # team_values = team_values[np.where(team_values[:,0] >= 38.33333)]
    # team_values = team_values[np.where(team_values[:,0] <= 76.66667)]

    # Only look at passes of possessions that lead to shots
    # # team_values = data[t].getSlicedValues(['x', 'y','poss_pass','poss_shot'])
    # team_values = data[t].getSlicedValues(['x', 'y','poss_pass','poss_off_entry'])
    # # team_values = team_values[np.where(team_values[:,2] == 1)]
    # team_values = team_values[np.where(team_values[:,3] == 1)]
    # # print team_values.shape
    # team_data, n = random_split(team_values[:,[0,1]], num_splits=4)

    # team_data, n = random_split(team_values, num_splits=10)
    team_data, n = split_games(data[t])
    print team_data.shape

    for d in team_data:
      y_e, x_e = get_bins()
      y_e[-1] = 76
      x_e[-1] = 116
      counts, y_edges, x_edges = np.histogram2d(d[:, 1], d[:, 0], bins = get_bins(), normed = True)
      height = y_edges[1] - y_edges[0]
      width = x_edges[1] - x_edges[0]
      counts = counts * height * width

      x.append(np.ravel(counts))
    y += [int(t)] * n

    # Sanity Check
    # for i in range(n):
    #   rand_index = random.randint(0, len(teams) - 1)
    #   y.append(teams[rand_index])

  return np.array(x), np.array(y)

def get_team_histo_pass_features(data, teams):
  y = []; x = []

  for t in teams:
    team_values = data[t].getSlicedValues(['distance', 'outcome'])
    team_data, n = random_sampling(team_values)
    for d in team_data:
      all_passes = get_histo(d[:,0])
      success = d[(d[:, 1] == 1), 0]
      successful_passes = get_histo(success)
      feature = np.concatenate((np.ravel(all_passes), np.ravel(successful_passes)))
      x.append(feature)
    y += [int(t)] * n
  return np.array(x), np.array(y)

def knn_classify(x_train, x_test, y_train, y_test, k = 5):
  clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
  clf.fit(x_train, y_train)
  x_predict = clf.predict(x_test)
  score = clf.score(x_test, y_test)
  accuracy = calc_accuracy(x_predict, y_test)
  print score, accuracy
  return x_predict

def knn_classification(event, seasons):
  """ Randomly resamples with replacement a season's worth of events for each team. Calculates the 2d histogram of each sample to create a feature vector,
  then performs knn classification """ 
  # Get Teams, Data
  print 'Getting data'
  data_teams = getAllTeamValues(event, seasons)
  teams = []
  for s in seasons:
    teams += GameDataPoss.getSeasonTeamIDs(s)
    
  print 'Splitting data'
  X, y = get_team_histo2d_features(data_teams, teams)
  # X,y = get_team_histo_pass_features(data_teams, teams)
  print X.shape
  print y.shape
  X_train, X_test, y_train, y_test = create_test_split(X, y, test_size=0.3)

  # Number of Neighbours
  print 'Classifying'
  x_predict = knn_classify(X_train, X_test, y_train, y_test, k = 5)
  return X, y, x_predict, X_test, y_test, X_train, y_train



##################################################### Experiments ############################################################

def knn_classification_experiment(season, NUM_EXPERIMENTS = 200):
  np.random.seed(0)

  data_teams = getAllTeamValues('pass', [season])
  teams = GameDataPoss.getSeasonTeamIDs(season)

  num_passes = {}
  total_passes = 0
  for t in teams:
    team_values = data_teams[t]
    team_values = team_values.getSlicedValues(['x', 'y'])
    num_passes[t] = team_values.shape[0]
    team_values = team_values[np.where(team_values[:,0] >= 38.33333)]
    team_values = team_values[np.where(team_values[:,0] <= 76.66667)]
    num_passes[t] = (num_passes[t] - team_values.shape[0]) / float(num_passes[t])
    # total_passes += team_values.shape[0]
  print num_passes
  return num_passes
  

  accuracy = []
  confusion = []
  precision = []
  recall = []
  k = []

  tuned_params = [{'n_neighbors': [2, 3, 4, 5, 6, 7]}]
  # tuned_params = [{'n_neighbors': [1, 2, 3]}]

  for i in range(10):

    X, y = get_team_histo2d_features(data_teams, teams)

    for i in range(NUM_EXPERIMENTS):
      # print '-------------Experiment #', i, '-----------------'
      X_train, X_test, y_train, y_test = create_test_split(X, y, test_size=0.3)

      clf = GridSearchCV(KNeighborsClassifier(weights='distance'), param_grid = tuned_params, cv = cross_validation.StratifiedKFold(y_train))
      # clf = GridSearchCV(KNeighborsClassifier(), param_grid = tuned_params)
      clf.fit(X_train, y_train)

      y_true, y_pred = y_test, clf.predict(X_test)
      acc = clf.score(X_test, y_test)
      # print(classification_report(y_true, y_pred))
      # print acc
      # print clf.grid_scores_
      # print clf.best_params_
      # print confusion_matrix(y_true, y_pred)
      p = precision_score(y_true, y_pred, average=None)
      r = recall_score(y_true, y_pred, average=None)
      precision.append(p)
      recall.append(r)
      k.append(clf.best_params_['n_neighbors'])


      accuracy.append(acc)
      confusion.append(confusion_matrix(y_true, y_pred))

  c = confusion[0]
  for c_matrix in confusion[1:]:
    c = c + c_matrix
  c = c / float(NUM_EXPERIMENTS * 10)


  # saveArrayAsCsv(c, season + '_midfield_confusion_matrix.txt')

  return np.array(accuracy), c, np.array(precision), np.array(recall), np.array(k)
    
def distances_experiment(season, event, NUM_EXPERIMENTS = 10, NUM_SAMPLES = 10):
  """ Creates NUM_SAMPLES per team. Creates the features for each sample, and constructs the distance matrix between all samples. Computes 
  the average distance between the samples for each pair of teams (including itself). Repeats the experiment NUM_EXPERIMENTS times, 
  and takes the average distance. Returns the averaged value of the average distances between each pair of teams """
  random.seed(0)
  data_teams = getAllTeamValues(event, [season])
  teams = GameDataPoss.getSeasonTeamIDs(season)
  num_teams = len(teams)

  averages = np.zeros((num_teams, num_teams))

  for i in xrange(NUM_EXPERIMENTS):
    X, y = get_team_histo2d_features(data_teams, teams)
    dist_mat = create_distance_matrix(X)
    a = get_average_distances_between_teams(dist_mat, num_teams, NUM_SAMPLES)
    averages = averages + a

  averages = averages / float(NUM_EXPERIMENTS)
  saveArrayAsCsv(averages, season + '-average-distances.txt', precision = 10)
  return averages

def clustering_experiment(season, event, NUM_EXPERIMENTS = 10, NUM_SAMPLES = 10):
  """ Creats NUM_SAMPLES per team. Creates the feature for each sample, and performs K-Means clustering """
  
  data_teams = getAllTeamValues(event, [season])
  teams = GameDataPoss.getSeasonTeamIDs(season)
  num_teams = len(teams)

  # for i in xrange(NUM_EXPERIMENTS):
  X, y = get_team_histo2d_features(data_teams, teams)
  est = KMeans(n_clusters = num_teams)
  labels = est.fit_predict(X)
  label_dict = {}

  for i in xrange(len(labels)):
    if labels[i] not in label_dict:
      label_dict[labels[i]] = [y[i]]
    else:
      label_dict[labels[i]].append(y[i])

  
  return labels, y, label_dict









###################################################### Clustering ##################################################
def clusterTest(values):
  print 'Calculating Distance'
  data_dist = pdist(values)
  print 'Calculating Linkages'
  data_link = linkage(data_dist)
  dendrogram(data_link)
  plt.xlabel('Samples')
  plt.ylabel('Distance')
  plt.show()


#################################################### Specific Plots ################################################

def plotPassesHisto(passes, s = 'epl13', save = False):
  total = passes.getValues('distance')
  success = total[passes.getValues('outcome') == 1]
  unsuccessful = total[passes.getValues('outcome') == 0]
  data = [total, success, unsuccessful]
  labels = ['All', 'Successful', 'Unsuccessful']
  if save:
    multiHisto(data, labels, numBins = 20, title = "Distances of Passes for " + s, xlabel = "Approximate Distance in yards", save = True, filename = s + '_pass_hist.png')
  else:
    multiHisto(data, labels, numBins = 20, title = "Distances of Passes for " + s, xlabel = "Approximate Distance in yards")

def plot_pass_histo_multiple_seasons(seasons = ['epl13', 'lgue13', 'liga13', 'ser13', 'bnds13']):
  e = 'pass'
  print 'plotting passes'
  for s in seasons:
    print 'Getting seasons ' + s
    v = getSeasonsValues(e, [s])
    plotPassesHisto(v, s = s, save = True)

def plot_histo2d_multiple_seasons(event, seasons = ['epl13', 'lgue13', 'liga13', 'ser13', 'bnds13'], goals = False, outcome = False):
  print 'plotting hist2d'
  for s in seasons:
    print 'Getting seasons ' + s
    v = getSeasonsValues(event, [s])
    data = (v.getValues('x'), v.getValues('y'))
    # # d = v.getSlicedValues(['x', 'y','poss_pass','poss_shot'])
    # d = v.getSlicedValues(['x', 'y','poss_pass','poss_off_entry'])
    # # print d
    # # d = d[np.where(d[:,2] == 1)]
    # d = d[np.where(d[:,3] == 1)]
    # data = [d[:,0], d[:,1]]
    # print data

    fname = s + '_' + event + '.eps'
    histo2d(data, ranged = False, title = s + ' ' + event.capitalize() + ' frequencies', save = True, filename = fname)

def plot_histo2d_all_teams(event, season, goals = False, outcome = False):
  print 'plotting hist2d for teams'
  teams = GameDataPoss.getSeasonTeamIDs(season)
  for t in teams:
    print 'Getting team', t
    v = getTeamValues(event, t, season)
    data = (v.getValues('x'), v.getValues('y'))
    fname = t + '_' + event + '-hist2d.png'
    histo2d(data, ranged = False, title = event.capitalize() + ' frequencies for team ' + t, numBins = get_bins(), save = True, filename = fname)

def plot_histo_passes_all_teams(season, save = False):
  print 'plotting hist for all teams'
  teams = GameDataPoss.getSeasonTeamIDs(season)
  for t in teams:
    print 'Team', t
    v = getTeamValues('pass', t, season)
    all_distances = v.getValues('distance')
    outcomes = v.getValues('outcome')
    success = all_distances[outcomes == 1]
    data = [all_distances, success]
    labels = ['All', 'Successful']
    multiHisto(data, labels, numBins = 20, title = 'Distances of Passes for ' + str(t), xlabel = 'Approximate Distance in yards')


#################################################### Main Procedure ################################################

def mainProcedure():
  # events = ['pass', 'shot', 'tackle']
  # plot_pass_histo_multiple_seasons()
  # for e in events:
  #   plot_histo2d_multiple_seasons(e)

  # Read Events
  # Split Events
  # Convert Events into Matrix of values
  print 'Getting Data'
  # s = ["epl13"]
  s = 'epl13'
  event = 'pass'
  # values = getSeasonsValues(event, s)
  values = getTeamValues(event, '11', s)

  # Transform Coordinates - Should be done in initialisation of values
  # values.transformCoordinates()
  # values.updateValues('distance', scaleDistances(values.getValues('x'), values.getValues('y'), values.getValues('end_x'), values.getValues('end_y')))

  # Remove duplicates (i.e. Aerial is repeated for each team)
  # Duplicates are: Aerial, Foul
  # values = values[0::2]


  # Heat Map Exploration
  data = (values.getValues('x'), values.getValues('y'))
  # weights = values.data[:,2]
  # onTarget = values.data[:,9]
  # goal = values.data[:,10]
  # histogram(data[0])
  # histogram(data[1])
  # histogram(values.data[:,0], weights = weights)
  # histogram(values.data[:,1], weights = weights)
  # hexHeatmap(data)
  histo2d(data, ranged = False, numBins = (20, 15))
  # plotPassesHisto(values)
  return values
  # histo2d(data, weights = weights)
  # histo2d(data, weights = onTarget)
  # histo2d(data, weights = goal)

############################################# Utils #######################################################

def saveArrayAsCsv(a, fname, precision = 1):
  np.savetxt('Clustering/Data/paper/' + fname, a, fmt = '%.' + str(precision) +'f', delimiter=',')
