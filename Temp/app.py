import streamlit as str
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import ezdxf
import io

# Set page title and layout
str.set_page_config(page_title="TIN-Guided Boundary Extractor", layout="wide")
str.title("🗺️ TIN-Guided Boundary Extractor & DXF Generator")
str.markdown("Extracts outer boundary layers from raw geodetic point datasets using spatial triangulation guidance.")

# --- HELPER FUNCTIONS FOR GEOMETRY ---
def parse_raw_text(text_data):
    """Parses whitespace/tab-separated X Y Z text data safely."""
    points = []
    for line in text_data.strip().split('\n'):
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2]) if len(parts) > 2 else 0.0
                points.append([x, y, z])
            except ValueError:
                continue
    return np.array(points)

def extract_tin_boundary(pts_2d):
    """
    Constructs a TIN mesh to safely identify localized neighborhood sets (T).
    The true outer boundary consists of edges belonging to exactly ONE triangle.
    This replaces the brute-force O(n^3) triangle search with a solid O(n log n) framework.
    """
    tri = Delaunay(pts_2d)
    edge_registry = {}
    
    # Track the occurrence frequency of every undirected edge
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            edge = tuple(sorted((u, v)))
            edge_registry[edge] = edge_registry.get(edge, 0) + 1
            
    # Filter out boundary edges (edges belonging to exactly 1 triangle)
    boundary_edges = [edge for edge, count in edge_registry.items() if count == 1]
    
    # Build an adjacency graph mapping for boundary walk tracking
    adj_graph = {}
    for u, v in boundary_edges:
        if u not in adj_graph: adj_graph[u] = []
        if v not in adj_graph: adj_graph[v] = []
        adj_graph[u].append(v)
        adj_graph[v].append(u)
        
    if not adj_graph:
        return []
        
    # Start tracing from X_min as defined in your prompt specifications
    x_min_idx = np.argmin(pts_2d[:, 0])
    
    boundary_ordered_indices = []
    current_node = x_min_idx
    prev_node = None
    
    # Traverse sequentially until returning safely to X_min
    while True:
        boundary_ordered_indices.append(current_node)
        neighbors = adj_graph.get(current_node, [])
        
        if len(boundary_ordered_indices) > 1 and current_node == x_min_idx:
            break
            
        # Select the next node along the perimeter path
        next_node = None
        for n in neighbors:
            if n != prev_node:
                # If returning home is possible, prioritize closing the loop safely
                if n == x_min_idx and len(boundary_ordered_indices) > 2:
                    next_node = n
                    break
                if n not in boundary_ordered_indices:
                    next_node = n
                    break
                    
        if next_node is None:
            # Fallback path closure catch-all
            if x_min_idx in neighbors and len(boundary_ordered_indices) > 2:
                boundary_ordered_indices.append(x_min_idx)
            break
            
        prev_node = current_node
        current_node = next_node
        
    return boundary_ordered_indices

def compute_polygon_properties(coords):
    """Computes total perimeter length and polygon area using the Shoelace formula."""
    x = coords[:, 0]
    y = coords[:, 1]
    # Perimeter
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    perimeter = np.sum(np.sqrt(dx**2 + dy**2))
    # Area
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return perimeter, area

def build_dxf(points, boundary_indices):
    """Generates standard dual-layer DXF CAD assets matching engineering specs."""
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Layer Definitions
    doc.layers.new(name='CAODO', color=3)     # Green text layers
    doc.layers.new(name='BOUNDARY', color=1)  # Red outer lines
    
    # Write Point elevation labels (TEXT elements)
    for p in points:
        msp.add_text(
            text=f"{p[2]:.3f}",
            dxfattribs={
                'layer': 'CAODO',
                'height': 0.4,
                'insert': (p[0], p[1], p[2]),
                'style': 'Standard'
            }
        )
        
    # Draw Closed Boundary Polyline
    if len(boundary_indices) > 2:
        poly_points = [(points[idx][0], points[idx][1], points[idx][2]) for idx in boundary_indices]
        # Guarantee closure configuration
        if poly_points[0] != poly_points[-1]:
            poly_points.append(poly_points[0])
            
        msp.add_lwpolyline(
            points=[(p[0], p[1]) for p in poly_points], 
            dxfattribs={'layer': 'BOUNDARY', 'flags': 1} # flags=1 implies Closed Polyline
        )
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()

# --- DEFAULT EMBEDDED GEODETIC RAW DATASET ---
DEFAULT_RAW_DATA = """558154.823	1153935.014	0.679
558153.646	1153940.001	0.54
558153.714	1153945.897	0.943
55814.105	1153945.305	0.52
558162.959	1153946.233	0.438
558164.454	1153941.043	0.364
558164.967	1153936.894	0.407
558175.643	1153937.452	0.469
558175.997	1153942.73	0.359
558175.84	1153947.323	0.416
558186.623	1153948.234	0.4
558187.765	1153943.912	0.267
558187.828	1153938.039	0.489
558198.218	1153939.243	0.464
558198.79	1153944.351	0.291
558198.898	1153949.351	0.303
558210.47	1153950.172	0.249
558212.052	1153945.455	-0.065
558212.777	1153940.2	-0.014
558224.025	1153941.162	-0.076
558224.745	1153945.329	-0.234
558225.592	1153951.072	-0.021
558226.25	1153945.936	-0.167
558226.103	1153941.25	-0.118
558235.144	1153941.756	-0.165
558235.336	1153946.902	-0.259
558234.698	1153951.394	-0.184
558242.966	1153952.174	-0.19
558244.191	1153947.242	-0.526
558248.471	1153947.009	-0.378
558247.93	1153942.549	-0.393
558243.668	1153942.149	-0.252
558243.427	1153945.508	-0.653
558249.775	1153942.704	-0.285
558256.373	1153943.142	-0.194
558262.873	1153943.46	-0.25
558272.791	1153943.854	-0.204
558280.225	1153944.5	-0.2
558291.524	1153944.777	-0.091
558300.373	1153945.795	-0.214
558309.886	1153946.296	-0.24
558322.085	1153947.188	0.187
558332.397	1153947.898	0.291
558339.661	1153948.105	0.287
558345.956	1153948.479	0.318
558351.404	1153948.658	0.393
558359.918	1153949.178	0.522
558368.312	1153949.641	0.5
558366.008	1153954.552	0.296
558367.884	1153954.914	0.458
558366.703	1153961.629	0.593
558366.362	1153961.093	0.502
558356.717	1153960.56	0.472
558356.171	1153954.785	0.412
558347.614	1153954.007	0.314
558345.77	1153959.554	0.423
558334.772	1153959.108	0.354
558333.764	1153953.33	0.167
558323.538	1153952.769	0.195
558320.871	1153958.467	0.255
558313.658	1153953.285	0.084
558311.947	1153957.88	0.17
558305.356	1153957.432	-0.128
558298.227	1153957.019	-0.047
558288.745	1153956.246	-0.109
558277.91	1153955.484	-0.073
558266.434	1153954.31	0.017
558250.001	1153952.777	-0.035
558241.464	1153952.073	-0.034
558229.066	1153950.927	0.001
558215.796	1153950.39	0.1"""

# --- APPLICATION INTERFACE SIDEBAR ---
str.sidebar.header("📁 Data Management Options")
data_source = str.sidebar.radio("Data Import Source:", ("Use Provided Sample Dataset", "Upload Custom TXT File"))

raw_text_content = DEFAULT_RAW_DATA
if data_source == "Upload Custom TXT File":
    uploaded_file = str.sidebar.file_uploader("Upload geometry coordinate text file (X Y Z format)", type=["txt", "csv"])
    if uploaded_file is not None:
        raw_text_content = uploaded_file.read().decode("utf-8")

# Parse string to working array
parsed_points = parse_raw_text(raw_text_content)

if len(parsed_points) < 3:
    str.error("❌ Need a minimum of 3 spatial points to evaluate boundary paths.")
else:
    # --- PROCESSING PIPELINE ---
    pts_2d = parsed_points[:, :2]
    boundary_indices = extract_tin_boundary(pts_2d)
    
    # Calculate geometric values
    boundary_coords = pts_2d[boundary_indices]
    total_perimeter, closed_area = compute_polygon_properties(boundary_coords)
    
    # --- METRICS INDICATORS DISPLAY ---
    col1, col2, col3 = str.columns(3)
    col1.metric("📊 Total Sample Points", len(parsed_points))
    col2.metric("📏 Perimeter Length", f"{total_perimeter:.3f} m")
    col3.metric("📐 Enclosed Surface Area", f"{closed_area:.3f} m²")
    
    # --- DATA PRESENTATION VIEWPORTS ---
    view_tab, data_tab = str.tabs(["👁️ Interactive Geometry View", "📋 Parsed Raw Table Data"])
    
    with view_tab:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Plot full internal TIN triangulation wireframe transparently for visual alignment
        tri = Delaunay(pts_2d)
        ax.triplot(pts_2d[:, 0], pts_2d[:, 1], tri.simplices, color='#CCCCCC', linewidth=0.6, linestyle='--', label='TIN Internal Links')
        
        # Plot geodetic target nodes
        ax.scatter(pts_2d[:, 0], pts_2d[:, 1], color='#1f77b4', s=18, zorder=3, label='Geodetic Points')
        
        # Overlay Elevation value text offset strings
        for p in parsed_points:
            ax.text(p[0] + 0.3, p[1] + 0.3, f"{p[2]:.2f}", fontsize=7, color='#2c3e50', alpha=0.85)
            
        # Highlight boundary nodes and closed path lines
        if len(boundary_indices) > 0:
            closed_loop_idx = boundary_indices + [boundary_indices[0]]
            ax.plot(pts_2d[closed_loop_idx, 0], pts_2d[closed_loop_idx, 1], color='#e74c3c', linewidth=2, zorder=4, label='Calculated Boundary Loop')
            
            # Anchor start position indicator label
            x_min_pt = pts_2d[boundary_indices[0]]
