import networkx as nx
import sqlite3 as sq
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
from pathlib import Path

db_path = Path(__file__).with_name("baseline.db")
conn = sq.connect(db_path)
G = nx.DiGraph()

conn.row_factory = sq.Row

node_rows = conn.execute(
'''
SELECT
model.model_id,
model.equation,
model.equation_family,
paper.authors,
paper.year
FROM model JOIN paper
    ON model.paper_id = paper.paper_id;
'''
)

for row in node_rows:
    node_label = (f"{row["authors"]} {row["year"]}\n" 
                  f"equation {row["equation"]}")

    G.add_node(
        row["model_id"],
        label = node_label,
        equation = row["equation"],
        equation_family = row["equation_family"]
    )

edge_rows = conn.execute(
'''
SELECT
relationship.parent_model_id,
relationship.child_model_id,
relationship.relation
FROM relationship;
'''
)

for row in edge_rows:
    G.add_edge(
        row["parent_model_id"],
        row["child_model_id"],
        relation = row["relation"],
        label = row["relation"]
    )


nx.draw(
    G,
    with_labels = True,
    arrows = True
)

dot_path = Path(__file__).with_name("baseline.dot")
write_dot(G, dot_path)



plt.show()
