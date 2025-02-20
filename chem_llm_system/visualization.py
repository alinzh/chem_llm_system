from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from chain import app

img_data = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

with open("graph.png", "wb") as f:
    f.write(img_data) 
