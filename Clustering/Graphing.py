 # -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import GameDataPoss

DATA_PATH = 'Clustering/Data/paper/'
FIG_PATH = 'Clustering/Figures/'

season_standing = [178, 186, 175, 188, 191, 182, 185, 184, 179, 1450,
              855, 174, 177, 192, 5683, 450, 176, 181, 180, 190]

graph_order = [175, 180, 174, 182, 179, 1450, 177, 181, 184, 5683, 
                855, 191, 185, 190, 176, 450, 192, 186, 188, 178]

def get_team_dict(season):
  team_dict = {}
  f = open(DATA_PATH + season + '-teams.txt', 'r')
  for l in f:
    l = l.split(',')
    team_dict[l[0]] = l[1][:-1]
  return team_dict


def create_distance_graph(teams, distances):
  ''' Create a graph using the distances between teams as edges, and teams as nodes'''

  num_teams = len(teams)

  G = nx.Graph()
  G.add_nodes_from(teams)

  for i in xrange(num_teams):
    for j in xrange(i + 1):
      G.add_edge(teams[i], teams[j], weight=distances[i][j])

  return G

def coloured_edge_graph(teams, distances, teams_dict):
  ''' Create a graph. The edge colour is dependent on the distance between the nodes. Teams are nodes '''
  num_teams = len(teams)
  G = nx.Graph()

  distance_values = []
  for i in xrange(num_teams):
    for j in xrange(i + 1):
      distance_values.append(distances[i][j])

  distance_values = sorted(distance_values)

  # # normalizing distance values
  # # distance_values

  num_distances = len(distance_values)
  first_quartile, second_quartile, third_quartile = distance_values[num_distances / 10], distance_values[num_distances / 2], distance_values[9 * num_distances / 10]
  # print first_quartile, second_quartile, third_quartile
  # normalize distances
  # distances = (distances - np.mean(distances)) / np.std(distances)

  # scale distances to 0 - 100
  # old_max = np.max(distances)
  # old_min = np.min(distances)
  # distances = 100.0 / (old_max - old_min) * (distances - old_min)

  for i in xrange(num_teams):
    # G.add_node(teams_dict[teams[i]], weight = distances[i][i])
    # G.add_node(teams[i], weight = distances[i][i])
    d = distances[i][i]
    if d <= first_quartile:
      G.add_node(teams_dict[teams[i]], {'category': 'first_quartile'})
    elif d <= second_quartile:
      G.add_node(teams_dict[teams[i]], {'category': 'second_quartile'})
    elif d <= third_quartile:
      G.add_node(teams_dict[teams[i]], {'category': 'third_quartile'})
    else:
      G.add_node(teams_dict[teams[i]], {'category': 'fourth_quartile'})

  for i in xrange(num_teams):
    for j in xrange(i + 1):
      if i != j:
        # G.add_edge(teams_dict[teams[i]], teams_dict[teams[j]], weight = distances[i][j])
        # G.add_edge(teams[i], teams[j], weight = distances[i][j])
        d = distances[i][j]
        if d <= first_quartile:
          G.add_edge(teams_dict[teams[i]], teams_dict[teams[j]], {'category': 'first_quartile'})
        elif d <= second_quartile:
          G.add_edge(teams_dict[teams[i]], teams_dict[teams[j]], {'category': 'second_quartile'})
        elif d <= third_quartile:
          G.add_edge(teams_dict[teams[i]], teams_dict[teams[j]], {'category': 'third_quartile'})
        else:
          G.add_edge(teams_dict[teams[i]], teams_dict[teams[j]], {'category': 'fourth_quartile'})

  return G

def get_position(G, team_dict):
  pos = nx.circular_layout(G)
  print pos
  new_pos = {}

  for i in range(len(season_standing)):
    team_std = team_dict[str(season_standing[i])]
    team_graph = team_dict[str(graph_order[i])]
    new_pos[team_std] = pos[team_graph]

  return new_pos


def draw_colored_graph(G, pos):
  color_map = {'first_quartile': '#FF0000', 'second_quartile': '#FF69B4', 'third_quartile': '#00BFFF', 'fourth_quartile': '#0000FF'}

  # edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
  # nodes,n_weights = zip(*nx.get_node_attributes(G,'weight').items())

  node_colors = [color_map[G.node[node]['category']] for node in G.nodes()]
  edge_colors = []
  edge_list = []
  for e in G.edges():
    # Only draw edges in the first or fourth quartile
    cat = G.get_edge_data(e[0], e[1])['category']
    if cat == 'first_quartile' or cat == 'fourth_quartile':
      edge_colors.append(color_map[cat])
      edge_list.append(e)

  print pos

  plt.figure()
  # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues)
  nx.draw(G, pos, node_color=node_colors, edge_color=edge_colors, edgelist=edge_list)

  labels = {}
  for n in G.nodes():
    labels[n] = n
  for node in labels:
    p = pos[node].copy()
    if p[0] >= 0.49:
      p[0] += 0.01
    else:
      p[0] -= 0.12
    if p[1] >= 0.49:
      p[1] += 0.04
    else:
      p[1] -= 0.055
    if labels[node] == 'Deportivo de La Coruna':
      plt.annotate('Dep. La Coruna', xy=p, size = 9)    
    else:
      plt.annotate(labels[node], xy=p, size = 9)  

  plt.axis('off')
  plt.savefig(FIG_PATH + 'exp3_sparse_ordered.eps', format='eps', dpi=800)
  # plt.show()

  # plt.show()


def draw_graph(G):
  ''' Draws the given graph G '''

  pos = nx.spring_layout(G)
  # pos=nx.graphviz_layout(G,prog="neato")
  nx.draw_networkx_nodes(G, pos, node_size = 700)
  nx.draw_networkx_edges(G, pos)
  labels = {}
  for n in G.nodes():
    labels[n] = n
  for node in labels:
    p = pos[node]
    if p[0] >= 0.49:
      p[0] += 0.05
    else:
      p[0] -= 0.1
    if p[1] >= 0.49:
      p[1] += 0.05
    else:
      p[1] -= 0.05
    plt.annotate(labels[node], xy=p) 

  plt.axis('off')
  plt.show()

def draw_season_graph(season='liga12'):
  fname = DATA_PATH + season + '-average-distances.txt'
  distances = np.loadtxt(fname, delimiter=',', dtype='float')
  teams = GameDataPoss.getSeasonTeamIDs(season)
  team_dict = get_team_dict(season)


  G = coloured_edge_graph(teams, distances, team_dict)

  # G = create_distance_graph(teams, distances)

  pos = get_position(G, team_dict)

  draw_colored_graph(G, pos)
  return G





