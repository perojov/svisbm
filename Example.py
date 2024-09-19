import networkx as nx
from BernoulliModel import BernoulliModelState
from VariationalInference import VariationalInference
from Utilities import train_test_split

edgelist = open("Graphs/Cit-HepPh.txt", "r")
G = nx.parse_edgelist(edgelist, delimiter=',', create_using=nx.Graph(), nodetype=int)
G = nx.convert_node_labels_to_integers(G)
X, sampled_edges = train_test_split(G)
state = BernoulliModelState(X, 200, sampled_edges)
vi = VariationalInference(state, 30)
vi.run()
#state = BernoulliModelStochasticState(X, 50, 20000)
#svi = StochasticVariationalInference(state)
#svi.run()
