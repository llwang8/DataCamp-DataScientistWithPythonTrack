# DataCamp
# Network Analysis in Python

# Exercises

# 1. Intro to Networks

# Basics of NetworkX API, using Twitter network
# What is the size of the graph T,
# ? the type of T.nodes(),
# ? the data structure of the third element of the last entry of
# T.edges(data=True)?
len(T)
# 23369
type(T.nodes())
#list
T.edges(data=True)[-1]
# (23324, 23327, {'date': datetime.date(2008, 2, 9)})

# Basic drawing of a network using NetworkX
# Import necessary modules
import matplotlib.pyplot as plt
import networkx as nx
# Draw the graph to screen
nx.draw(T_sub)
plt.show()

# Queries on a graph
# Use a list comprehension to get the nodes of interest: noi
noi = [n for n, d in T.nodes(data=True) if d['occupation'] == 'scientist']
# Use a list comprehension to get the edges of interest: eoi
eoi = [(u, v) for u, v, d in T.edges(data=True) if d['date'] < date(2010, 1, 1)]


# specifying a weight on edges
# Set the weight of the edge
weight = 2
# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):
    # Check if node 293 is involved
    if 293 in T.edges(data=True):
        # Set the weight to 1.1
        d['weight'] = 1.1


# Set the weight of the edge
T.edge[1][10]['weight'] = 2
# Iterate over all the edges (with metadata)
for u, v, d in T.edges(data=True):
    # Check if node 293 is involved
    if 293 in [u, v]:
        # Set the weight to 1.1
        T[u][v]['weight'] = 1.1


# Checking whether there are self-loops in the graph
# Define find_selfloop_nodes()
def find_selfloop_nodes(G):
    """
    Finds all nodes that have self-loops in the graph G.
    """
    nodes_in_selfloops = []
    # Iterate over all the edges of G
    for u, v in G.edges():
    # Check if node u and node v are the same
        if u == v:
            # Append node u to nodes_in_selfloops
            nodes_in_selfloops.append(u)
    return nodes_in_selfloops
# Check whether number of self loops equals the number of nodes in self loops
assert T.number_of_selfloops() == len(find_selfloop_nodes(T))


#Visualizing using Matrix plots
# Import nxviz
import nxviz as nv
# Create the MatrixPlot object: m
m = nv.MatrixPlot(T)
# Draw m to the screen
m.draw()
# Display the plot
plt.show()
# Convert T to a matrix format: A
A = nx.to_numpy_matrix(T)
# Convert A back to the NetworkX form as a directed graph: T_conv
T_conv = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
# Check that the `category` metadata field is lost from each node
for n, d in T_conv.nodes(data=True):
    assert 'category' not in d.keys()


# Visualizing using circos plots
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import CircosPlot
# Create the CircosPlot object: c
c = CircosPlot(T)
# Draw c to the screen
c.draw()

# Display the plot
plt.show()


# visualizing using Arc plots
# Import necessary modules
import matplotlib.pyplot as plt
from nxviz import ArcPlot
# Create the un-customized ArcPlot object: a
a = ArcPlot(T)
# Draw a to the screen
a.draw()
# Display the plot
plt.show()
# Create the customized ArcPlot object: a2
a2 = ArcPlot(T, node_order='category', node_color='category')
# Draw a2 to the screen
a2.draw()
# Display the plot
plt.show()


# 2. Important nodes

# compute number of neighbors for each node
# Define nodes_with_m_nbrs()
def nodes_with_m_nbrs(G, m):
    """
    Returns all nodes in graph G that have m neighbors.
    """
    nodes = set()
    # Iterate over all nodes in G
    for n in G.nodes():
        # Check if the number of neighbors of n matches m
        if len(G.neighbors(n)) == m:
            # Add the node n to the set
            nodes.add(n)
    # Return the nodes with m neighbors
    return nodes
# Compute and print all nodes in T that have 6 neighbors
six_nbrs = nodes_with_m_nbrs(T, 6)
print(six_nbrs)


# Compute degree distribution
# Compute the degree of every node: degrees
degrees = [len(T.neighbors(n)) for n in T.nodes()]
# Print the degrees
print(degrees)


# Degree centrality distribution
# Import matplotlib.pyplot
import matplotlib.pyplot as plt
# Compute the degree centrality of the Twitter network: deg_cent
deg_cent = nx.degree_centrality(T)
# Plot a histogram of the degree centrality distribution of the graph.
plt.figure()
plt.hist(list(deg_cent.values()))
plt.show()
# Plot a histogram of the degree distribution of the graph
plt.figure()
plt.hist(degrees)
plt.show()
# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure()
plt.scatter(degrees, list(deg_cent.values()))
plt.show()


# shortest Path

# Define path_exists()
def path_exists(G, node1, node2):
    """
    This function checks whether a path exists between two nodes (node1, node2) in graph G.
    """
    visited_nodes = set()
    # Initialize the queue of cells to visit with the first node: queue
    queue = [node1]
    # Iterate over the nodes in the queue
    for node in queue:
        # Get neighbors of the node
        neighbors = G.neighbors(node)
        # Check to see if the destination node is in the set of neighbors
        if node2 in neighbors:
            print('Path exists between nodes {0} and {1}'.format(node1, node2))
            return True
            break
        else:
            # Add current node to visited nodes
            visited_nodes.add(node)
            # Add neighbors of current node that have not yet been visited
            queue.extend([n for n in neighbors if n not in visited_nodes])
        # Check to see if the final element of the queue has been reached
        if node == queue[-1]:
            print('Path does not exist between nodes {0} and {1}'.format(node1, node2))
            # Place the appropriate return statement
            return False

# NetworkX betweenness centrality on a social network
# Compute the betweenness centrality of T: bet_cen
bet_cen = nx.betweenness_centrality(T)
# Compute the degree centrality of T: deg_cen
deg_cen = nx.degree_centrality(T)
# Create a scatter plot of betweenness centrality and degree centrality
plt.scatter(list(bet_cen.values()), list(deg_cen.values()) )
# Display the plot
plt.show()


# Deep Dive - Twitter newtwork

# Define find_nodes_with_highest_deg_cent()
def find_nodes_with_highest_deg_cent(G):
    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(G)
    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))
    nodes = set()
    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
        # Check if the current value has the maximum degree centrality
        if v == max_dc:
            # Add the current node to the set of nodes
            nodes.add(k)
    return nodes
# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc = find_nodes_with_highest_deg_cent(T)
print(top_dc)
# Write the assertion statement
for node in top_dc:
    assert nx.degree_centrality(T)[node] == max(nx.degree_centrality(T).values())


# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(G):
    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(G)
    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))
    nodes = set()
    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():
        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:
            # Add the current node to the set of nodes
            nodes.add(k)
    return nodes
# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc = find_node_with_highest_bet_cent(T)
# Write an assertion statement that checks that the node(s) is/are correctly identified.
for node in top_bc:
    assert nx.betweenness_centrality(T)[node] == max(nx.betweenness_centrality(T).values())


# 3. Structures

# Identifying triangle relationships
from itertools import combinations
# Define is_in_triangle()
def is_in_triangle(G, n):
    """
    Checks whether a node `n` in graph `G` is in a triangle relationship or not.
    Returns a boolean.
    """
    in_triangle = False
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if an edge exists between n1 and n2
        if G.has_edge(n1, n2):
            in_triangle = True
            break
    return in_triangle


# Finding modes involved in triangles
from itertools import combinations
# Write a function that identifies all nodes in a triangle relationship with a given node.
def nodes_in_triangle(G, n):
    """
    Returns the nodes in a graph `G` that are involved in a triangle relationship with the node `n`.
    """
    triangle_nodes = set([n])
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if n1 and n2 have an edge between them
        if G.has_edge(n1, n2):
            # Add n1 to triangle_nodes
            triangle_nodes.add(n1)
            # Add n2 to triangle_nodes
            triangle_nodes.add(n2)
    return triangle_nodes
# Write the assertion statement
assert len(nodes_in_triangle(T, 1)) == 35


# Finding open triangles
from itertools import combinations
# Define node_in_open_triangle()
def node_in_open_triangle(G, n):
    """
    Checks whether pairs of neighbors of node `n` in graph `G` are in an 'open triangle' relationship with node `n`.
    """
    in_open_triangle = False
    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):
        # Check if n1 and n2 do NOT have an edge between them
        if not G.has_edge(n1, n2):
            in_open_triangle = True
            break
    return in_open_triangle
# Compute the number of open triangles in T
num_open_triangles = 0
# Iterate over all the nodes in T
for n in T.nodes():
    # Check if the current node is in an open triangle
    if node_in_open_triangle(T, n):
        # Increment num_open_triangles
        num_open_triangles += 1
print(num_open_triangles)


# Finding all maximal cliques of size 'n'
# Define maximal_cliques()
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs
# Check that there are 33 maximal cliques of size 3 in the graph T
assert len(maximal_cliques(T, 3)) == 33


# Subgraphs

nodes_of_interest = [29, 38, 42]
# Define get_nodes_and_nbrs()
def get_nodes_and_nbrs(G, nodes_of_interest):
    """
    Returns a subgraph of the graph `G` with only the `nodes_of_interest` and their neighbors.
    """
    nodes_to_draw = []
    # Iterate over the nodes of interest
    for n in nodes_of_interest:
        # Append the nodes of interest to nodes_to_draw
        nodes_to_draw.append(n)
        # Iterate over all the neighbors of node n
        for nbr in G.neighbors(n):
            # Append the neighbors of n to nodes_to_draw
            nodes_to_draw.append(nbr)
    return G.subgraph(nodes_to_draw)
# Extract the subgraph with the nodes of interest: T_draw
T_draw = get_nodes_and_nbrs(T, nodes_of_interest)
# Draw the subgraph to the screen
nx.draw(T_draw)
plt.show()


# Extract the nodes of interest: nodes
nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']
# Create the set of nodes: nodeset
nodeset = set(nodes)
# Iterate over nodes
for n in nodes:
    # Compute the neighbors of n: nbrs
    nbrs = T.neighbors(n)
    # Compute the union of nodeset and nbrs: nodeset
    nodeset = nodeset.union(nbrs)
# Compute the subgraph using nodeset: T_sub
T_sub = T.subgraph(nodeset)
# Draw T_sub to the screen
nx.draw(T_sub)
plt.show()




# 4. Bringing it all together


