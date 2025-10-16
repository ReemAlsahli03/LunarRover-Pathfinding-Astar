import rasterio
from pyproj import CRS, Transformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import heapq
from math import sqrt

# Configuration
path = '/Users/reemyalfaisal/artemis/DM2_final_adj_5mpp_slp.tif'
#each grid represent 5x5 m on the moon
cell_size_m = 5.0
max_safe_slope = 15.0
#show animation of the pathfinding process
visualize_search_default = True
#Update the visualization every 200 iterations
vis_interval_default = 200
# wait 0.01 s between visualization frames updates
pause_time_default = 0.01
# maximum iterations to prevent infinite loops
max_iters_default = 300000
# ROI display a 50x50 cell area arund the current search location
graph_roi_size = 50
# limit the number of edges drawn in the ROI to avoid clutter, show nly last n explored connections.
recent_edges_limit = 200

# Open raster
src = rasterio.open(path)
slope = src.read(1).astype(float)
nodata = src.nodata #some cells might be nodata
crs = src.crs
if nodata is not None:
    slope[slope == nodata] = np.nan
n_rows, n_cols = slope.shape # get dimensions of the terrain grid

# -----Convert to coordinate system, get lat/long-----
# CRS transformers
moon_lonlat = CRS.from_proj4('+proj=longlat +R=1737400 +no_defs')
to_raster = Transformer.from_crs(moon_lonlat, crs, always_xy=True)
from_raster = Transformer.from_crs(crs, moon_lonlat, always_xy=True)



# --------- Cost grid ---------
#convert slope values (degrees) into cost (difficulty of traversal) 
slope_term = np.where(np.isnan(slope), np.nan, slope / max_safe_slope)
# divide each slope value by the max safe slope to get a normalized value between 0 and 1

cost_grid = np.where(np.isnan(slope), np.nan,
                     np.where(slope <= max_safe_slope, 1.0 + slope_term, np.inf))
# cost grid t create a cost for each cell, if slpe <= max_safe_slope, cost = 1 + normalized slope (between 1 and 2) 
# if slope > max_safe_slope, cost = infinity (impassable) 
# safe_mask boolean mask indicating safe cells
safe_mask = (slope <= max_safe_slope)

# Normalize for plotting 
# s_vmin == the 1st percentile of slope values (lowest), s_vmax == the 99th percentile of slope values (highest)
s_vmin, s_vmax = np.nanpercentile(slope, 1), np.nanpercentile(slope, 99)
c_display = cost_grid.copy()
c_display[np.isinf(c_display)] = np.nan
c_vmin, c_vmax = np.nanpercentile(c_display, 1), np.nanpercentile(c_display, 99)
# vmax == max value to display on the color scale, vmin == min value to display on the color scale 
norm_s = Normalize(vmin=s_vmin, vmax=s_vmax)
norm_c = Normalize(vmin=c_vmin, vmax=c_vmax)

# Figures
# creating two subplots side by side for slope and cost
fig, (ax_slope, ax_cost) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
im_s = ax_slope.imshow(slope, cmap='terrain', norm=norm_s, interpolation='nearest')
ax_slope.set_title(f'Slope (cells = {cell_size_m:.0f}×{cell_size_m:.0f} m)')
ax_slope.set_xlabel('Column'); ax_slope.set_ylabel('Row')
fig.colorbar(im_s, ax=ax_slope, fraction=0.046, pad=0.04, label='Slope (°)')

im_c = ax_cost.imshow(c_display, cmap='viridis', norm=norm_c, interpolation='nearest')
ax_cost.set_title('Traversal Cost (slope-only)')
ax_cost.set_xlabel('Column'); ax_cost.set_ylabel('Row')
fig.colorbar(im_c, ax=ax_cost, fraction=0.046, pad=0.04, label='Cost')

# Graph figure
fig_graph, ax_graph = plt.subplots(figsize=(6,6))
ax_graph.set_title("A* Graph Visualization (ROI)")
ax_graph.invert_yaxis()

# Interactive state
# start and end points for pathfinding
pointA = None; pointB = None
# visual markers for start and end points
markerA = None; markerB = None
# line for the current best path during visualization, and animate overlay during search
path_line = None; visual_overlay = None
# whether to show animation, set to True in line 15
visualize_search = visualize_search_default
running = True





# ----------- 8-connected neighbors --------------
def neighbors8(r,c):
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if dr==0 and dc==0: continue
            nr, nc = r+dr, c+dc
            if 0<=nr<n_rows and 0<=nc<n_cols:
                yield nr, nc, sqrt(dr*dr+dc*dc)
 # the 8 cells around the square, check for boundaries to make sure the neighbors are within the grid
 # calculate the distance to each neighbor (1 for orthogonal, sqrt(2) for diagonal)
 # different distance because mving diagonally is longer than moving horizontally or vertically. 


# r == current row, c == current column
# dr == change in row, dc == change in column
# nr == neighbor row (current row + dr), nc == neighbor column (current column + dc) 
# yield nr, nc, euclidian distance 





# --------- A* search with ROI graph -----------
# --- Variables names explained --- 
   # sr == start row, sc == start column
   # gr == goal row, gc == goal column
   # vis_interval == how often to update the visualization (as defined in line 17, set to 200)
   # max_iters == maximum iterations to prevent infinite loops 
   # slope_diff_weight == weight for slope difference penalty (to prefer smoother paths) 
   # finite_mask == boolean grid marking cells with valid costs
   # min_traversal == the cheapest cost in the entire grid
   # g_score == cost from start to current node
   # f == estimated total cost from start to goal through current node (g + heuristic)
   # open_heap == priority queue of (f, (row, col)) for nodes to explore, sorted by promising-ness 
   # came_from == dictionary to reconstruct the path
   # in_closed == boolean array to track nodes already evaluated
   # overlay == array for visualization overlay (0=unvisited, 1=open, 2=closed)
def astar(start, goal, visualize=True, pause_time=0.0, vis_interval=200,
          max_iters=300000, slope_diff_weight=0.02):
    sr, sc = start; gr, gc = goal
    if np.isnan(slope[sr, sc]) or np.isnan(slope[gr, gc]): print("Start/goal NaN"); return None, np.inf
    if np.isinf(cost_grid[sr, sc]) or np.isinf(cost_grid[gr, gc]): print("Start/goal impassable"); return None, np.inf

    finite_mask = np.isfinite(cost_grid)
    min_traversal = float(np.nanmin(cost_grid[finite_mask])) if np.any(finite_mask) else 1.0
    min_traversal = max(min_traversal, 1e-8)

    # heuristic = minimum cost * Euclidean distance 
    # a == point a , b == point b. [0] == row, [1] == column 
    def heuristic(a,b): return min_traversal*sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    # open set as a priority queue (min-heap) of (f, (row, col))
    open_heap = [(heuristic((sr, sc),(gr,gc)), (sr, sc))]
    # g_score = cost from start to current node
    g_score = np.full((n_rows, n_cols), np.inf); g_score[sr, sc]=0.0
    # came_from == dictionary to reconstruct the path
    came_from = {}; in_closed = np.zeros((n_rows,n_cols),dtype=bool)
    # grid for visualization overlay (0=unvisited, 2=closed)
    overlay = np.zeros((n_rows,n_cols),dtype=np.uint8)
    # counter tracker for how many steps A* has taken
    iter_count = 0
    # node_positions == list of all nodes added to the open set (for visualization)
    node_positions = []
    # edge_lines == list of edges (connections) explored (for visualization)
    edge_lines = []

    while open_heap:
        iter_count += 1
        if iter_count>max_iters: print("Max iters exceeded"); return None, np.inf
        _, (r,c) = heapq.heappop(open_heap) # heap pop, return the most promising node
        if in_closed[r,c]: continue
        in_closed[r,c]=True
        overlay[r,c]=2 # if the cell is explored, mark it as closed (2)
        node_positions.append((r,c)) 

        if (r,c)==(gr,gc): # if we reached the goal (gr, gc)
            path=[(gr,gc)] # a list of cells from start to goal
            while path[-1] in came_from: path.append(came_from[path[-1]]) # add cells by backtracking from goal to start
            path.reverse() # flip the list so it goes from start to goal

            print(f"A* finished: {len(path)} steps, g={g_score[gr,gc]:.3f}, iterations={iter_count}")
            plot_final_graph(came_from, g_score, path)
            return path, g_score[gr,gc] # return the path and the total cost, row and column of the goal

        for nr, nc, dist in neighbors8(r,c): # loop through each of the 8 neighbor cells
            # skip if either the neighbor or current cell is not traversable (NaN or inf) 
            if not np.isfinite(cost_grid[nr,nc]) or not np.isfinite(cost_grid[r,c]): continue 
            # average cost of current and neighbor cell, better than taking the maximum (it would overestimate)
            avg_trav = 0.5*(cost_grid[r,c]+cost_grid[nr,nc])
            # extra slope added for big slope changes, to prefer smoother paths
            slope_pen = slope_diff_weight*abs(slope[nr,nc]-slope[r,c])
            # the potential new cost to reach the neighbor through the current cell
            tentative_g = g_score[r,c]+dist*avg_trav+slope_pen
            if tentative_g < g_score[nr,nc]: # if this new path is better (lower cost) than any previous paths to this neighor
                g_score[nr,nc]=tentative_g # update the best known cost to reach this neighbor
                came_from[(nr,nc)] = (r,c) # remember where we came from
                f = tentative_g + heuristic((nr,nc),(gr,gc)) # calculate f score = g + h
                heapq.heappush(open_heap, (f,(nr,nc))) # add this neighbor to the exploration queue
                edge_lines.append(((c,r),(nc,nr),tentative_g)) # list of connections explored
                if len(edge_lines) > recent_edges_limit: edge_lines.pop(0) # keeping all edges would use too much memory, keep only recent 200 

        if visualize and iter_count % vis_interval == 0: # if visualization is enabled and it's time to update
            _visualize_overlay_graph(overlay, came_from, (r,c), node_positions, edge_lines)
            plt.pause(pause_time) 
    print("A* failed: no path found"); return None, np.inf

# Dynamic ROI visualization
def _visualize_overlay_graph(overlay, came_from, current, node_positions, edge_lines,
                             roi_size=50, max_edges=200):
    global visual_overlay, path_line
    # if first time, create the overlay image. if second time onward, just update it
    if visual_overlay is None: 
        cmap = plt.get_cmap('plasma')
        visual_overlay = ax_slope.imshow(overlay, cmap=cmap, alpha=0.45,
                                         interpolation='nearest', vmin=0, vmax=2)
    else:
        visual_overlay.set_data(overlay)

    # Draw current best path to current node
    path=[]
    node=current
    while node in came_from:
        path.append(node)
        node=came_from[node]
    path=path[::-1]
    xs=[c for r,c in path]; ys=[r for r,c in path] # matplotlib plot function needs separate lists for x and y
    if path_line is None:
        if len(xs)>1: path_line,=ax_slope.plot(xs,ys,color='yellow',linewidth=1.2)
    else:
        if len(xs)>1: path_line.set_data(xs,ys)
        else: path_line.set_data([],[])

    # Dynamic ROI around current node
    ax_graph.cla(); ax_graph.set_title("A* Graph ROI"); ax_graph.invert_yaxis()
    r0, c0 = current # row 0 column 0
    r_min, r_max = max(r0-roi_size//2, 0), min(r0+roi_size//2, n_rows) # min is bottom edge,, max is top edge
    c_min, c_max = max(c0-roi_size//2, 0), min(c0+roi_size//2, n_cols) # min is left edge, max is right edge
    ax_graph.set_xlim(c_min, c_max); ax_graph.set_ylim(r_min, r_max)

    # Nodes in ROI
    roi_nodes = [(r,c) for r,c in node_positions if r_min<=r<r_max and c_min<=c<c_max] # filters node_positions to only those within the ROI
    if roi_nodes:
        ax_graph.scatter([c for r,c in roi_nodes], [r for r,c in roi_nodes],
                         color='darkblue', s=20)

    # Only show the last max_edges edges
    recent_edges = edge_lines[-max_edges:] # last 200 edges explored
    for (x0,y0),(x1,y1),cost in recent_edges:
        if r_min<=y0<r_max and c_min<=x0<c_max and r_min<=y1<r_max and c_min<=x1<c_max:
            linewidth = max(0.5, min(cost/5.0, 2.5))
            ax_graph.plot([x0,x1],[y0,y1],color='darkgreen',linewidth=linewidth)

    #  draw partial path in ROI
    path_roi = [(r,c) for r,c in path if r_min<=r<r_max and c_min<=c<c_max]
    if len(path_roi)>1:
        ax_graph.plot([c for r,c in path_roi], [r for r,c in path_roi],
                      color='yellow', linewidth=2)

    fig.canvas.draw_idle(); fig_graph.canvas.draw_idle()


# Final graph with path, shows the complete search tree with all explored edges. 
def plot_final_graph(came_from, g_score, path):
    ax_graph.cla(); ax_graph.set_title("Final A* Graph"); ax_graph.invert_yaxis()
    # plot all explored edges
    for child,parent in came_from.items():
        x0,y0=parent[1],parent[0]; x1,y1=child[1],child[0] # convert x = column , y = row
        cost = g_score[child]-g_score[parent] # compute cost between parent and child
        ax_graph.plot([x0,x1],[y0,y1],color='darkgreen',linewidth=max(0.5,min(cost/5.0,2.5))) # line width based on cost, expensive edges thicker and cheap edges thinner. 
    # plot all explored nodes
    all_nodes = set(came_from.keys()) | set(came_from.values())
    ax_graph.scatter([n[1] for n in all_nodes],[n[0] for n in all_nodes],
                     color='darkblue', s=20)
    # highlight path
    path_x = [c for r,c in path]; path_y=[r for r,c in path]
    ax_graph.plot(path_x,path_y,color='red',linewidth=2) # plot path as bold red line 


    # edge costs, annotate cost between each pair of nodes in the path
    for i in range(1,len(path)):
        x0,y0=path[i-1][1],path[i-1][0]; x1,y1=path[i][1],path[i][0]
        mid_x, mid_y=(x0+x1)/2,(y0+y1)/2
        cost=g_score[path[i]]-g_score[path[i-1]] # cost between the two nodes
        ax_graph.text(mid_x,mid_y,f"{cost:.1f}",color='black',fontsize=6)
    fig_graph.canvas.draw_idle() # updates the graph without blocking 

# Click handlers
click_state = {'count':0}
def onclick_set_points(event):
    global pointA, pointB, markerA, markerB, path_line, visual_overlay
    if event.inaxes is not ax_slope: return
    if event.xdata is None or event.ydata is None: return
    col=int(event.xdata); row=int(event.ydata)
    if not (0<=row<n_rows and 0<=col<n_cols): return

    if click_state['count']%2==0: # to know if you're setting point A or B 
        pointA=(row,col)
        if markerA: markerA.remove()
        markerA=ax_slope.plot(col,row,marker='o',color='lime',markersize=8)[0] # Point A as green circle
        print(f"Point A set at row={row}, col={col}")
    else:
        pointB=(row,col)
        if markerB: markerB.remove()
        markerB=ax_slope.plot(col,row,marker='o',color='red',markersize=8)[0] # Point B as red circle
        print(f"Point B set at row={row}, col={col}")
       
        # clears the old animation before running pathfinding
        if visual_overlay: visual_overlay.remove(); visual_overlay=None
        if path_line: path_line.remove(); path_line=None

        # Run the A* 
        path, total_cost = astar(pointA,pointB,visualize=visualize_search,
                                 pause_time=pause_time_default, vis_interval=vis_interval_default,
                                 max_iters=max_iters_default)
        if path is not None: 
            cols=[c for r,c in path]; rows=[r for r,c in path]
            ax_slope.plot(cols,rows,color='yellow',linewidth=2) # Plot the path as yellow line
            fig.canvas.draw_idle()
            print(f"Final path length {len(path)}, total cost {total_cost:.3f}")
        else:
            print("No path found.")

    click_state['count']+=1
    fig.canvas.draw_idle()

# Keyboard
def on_key(event):
    global pointA, pointB, markerA, markerB, path_line, visual_overlay, visualize_search, running, click_state
    key=getattr(event,'key',None)
    if key=='r':
        pointA=pointB=None
        click_state['count']=0
        for obj in (markerA,markerB,path_line,visual_overlay):
            if obj: obj.remove()
        markerA=markerB=path_line=visual_overlay=None
        fig.canvas.draw_idle()
        print("Reset points and overlays.")


    elif key=='v':
        visualize_search = not visualize_search
        print(f"Visualization toggled -> {visualize_search}")


    elif key=='q':
        running=False
        plt.close('all')
        print("Quitting and closing windows.")

# Connect the keys and functions to be interactive 
cid_points = fig.canvas.mpl_connect('button_press_event', onclick_set_points)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)


# Keep the UI alive 
plt.ion(); plt.show(block=False)
try:
    while running and plt.get_fignums():
        plt.pause(0.1)
except KeyboardInterrupt:
    pass
finally:
    src.close()
