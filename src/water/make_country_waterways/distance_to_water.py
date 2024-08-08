import numpy as np
import rasterio as rio
from water.basic_functions import (ppaths, tt, time_elapsed, Path)
from water.make_country_waterways.cut_data import merge_data
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.shortest_paths import multi_source_dijkstra_path_length
from pyproj.geod import Geod


def make_waterway_population_grid(country: str,
                                  output_dir: str = 'output_data_cut',
                                  save_name: str = 'waterways_population_grid.tif'
                                  ):
    country_dir = ppaths.waterway/f'country_data/{country}'
    population_path = country_dir/'population.tif'
    model_output = list((country_dir/output_dir).glob('*'))
    merged_data = merge_data(model_output, population_path)
    merged_data = np.round(merged_data).astype(np.uint8)
    with rio.open(population_path) as src_f:
        profile = src_f.meta
        profile.update(
                {'count': 1,
                 'dtype': 'uint8',
                 'driver': 'GTiff',
                 'crs': src_f.crs,
                 'nodata': 0,
                 }
        )
        to_write = merged_data.to_numpy()
        with rio.open(country_dir/save_name, 'w', **profile) as dst_f:
            dst_f.write(to_write)



# def get_distance()


def add_edges_to_graph_edges(
        graph_edges: list,
        row: int,
        col: int,
        x_dist: float,
        y_dist: float,
        xy_dist: float
) -> list:
    edges = [
        ((row, col), (row + 1, col - 1), xy_dist),
        ((row, col), (row + 1, col), y_dist),
        ((row, col), (row + 1, col + 1), xy_dist),
        ((row, col), (row, col + 1), x_dist)
    ]
    graph_edges.extend(edges)
    return graph_edges


def get_distances(geod: Geod, row: int, xy):
    x1, y1 = xy(row, 0)
    x2, y2 = xy(row + 1, 1)
    x_dist = geod.line_length(lons=[x1, x2], lats=[y1, y1])
    y_dist = geod.line_length(lons=[x1, x1], lats=[y1, y2])
    xy_dist = geod.line_length(lons=[x1, x2], lats=[y1, y2])
    return x_dist, y_dist, xy_dist

def add_edges_to_graph(num_rows, num_cols, xy, graph):
    geod = Geod(ellps='WGS84')
    print('making edges')
    graph_edges = []
    s = tt()
    for row in range(num_rows - 1):
        x_dist, y_dist, xy_dist = get_distances(geod, row, xy)
        for col in range(num_cols - 1):
            graph_edges = add_edges_to_graph_edges(
                    graph_edges=graph_edges, row=row, col=col,
                    x_dist=x_dist, y_dist=y_dist, xy_dist=xy_dist
            )
    time_elapsed(s,2)
    print('adding edges')
    s = tt()
    graph.add_weighted_edges_from(graph_edges)
    time_elapsed(s, 2)
    return graph

def make_distance_grid(
        country: str, base_dir_path: Path = ppaths.waterway/f'country_data'
):
    country_dir = base_dir_path/f'{country}'
    water_pop_path = country_dir/'waterways_population_grid.tif'
    with rio.open(water_pop_path) as rio_f:
        grid = rio_f.read()[0]
        xy = rio_f.xy
    num_rows, num_cols = grid.shape
    graph = nx.Graph()
    graph = add_edges_to_graph(num_rows=num_rows, num_cols=num_cols, graph=graph, xy=xy)
    print('finding shortest paths')
    s = tt()
    sources = list(zip(*np.where(grid == 1)))
    distances = multi_source_dijkstra_path_length(G=graph, sources=sources)
    time_elapsed(s, 2)
    dist_grid = np.zeros(grid.shape, dtype=np.uint16)
    for (row, col), value in distances.items():
        dist_grid[row, col] = int(round(value))
    dist_grid = np.stack([dist_grid])
    with rio.open(water_pop_path) as src_f:
        profile = src_f.meta
        profile.update(
                {'count': 1,
                 'dtype': 'uint16',
                 'driver': 'GTiff',
                 'crs': src_f.crs,
                 'nodata': 0,
                 }
        )
        with rio.open(country_dir/'water_distance_grid.tif', 'w', **profile) as dst_f:
            dst_f.write(dist_grid)
    return dist_grid





if __name__ == '__main__':
    # make_waterway_population_grid('rwanda')
    country = 'rwanda'
    # dist = make_distance_grid(country)
    country_dir = ppaths.waterway/f'country_data/{country}'
    #
    with rio.open(country_dir/'waterways_population_grid.tif') as rio_ww:
        ww_data = rio_ww.read()[0]
    with rio.open(country_dir/'water_distance_grid.tif') as rio_wd:
        wd_data = rio_wd.read()[0]
    with rio.open(country_dir/'population.tif') as rio_pop:
        pop_data = rio_pop.read()[0]

    # ww_data[pop_data<0] = 0
    # wd_data[pop_data<0] = 0
    # pop_data[pop_data<0] = 0
    # grid = np.zeros(ww_data.shape, dtype=np.float32)
    # for (row, col), value in dist.items():
    #     grid[row, col] = value
    # with rio.open(country_dir/'population.tif') as rio_pop:
    #     pop_data = rio_pop.read()
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    plt.subplots_adjust(left=.025, bottom=.05, right=.975, top=.95, wspace=.01, hspace=.01)
    ax[0].imshow(ww_data, 'gray')
    ax[0].set_title('Water')
    ax[1].imshow(wd_data, cmap='inferno',
                 vmax=wd_data[wd_data > 0].mean() + 2*wd_data[wd_data > 0].std())
    ax[1].set_title('Distance to Water')
    ax[2].imshow(pop_data, vmin=-10,
                 vmax=pop_data[pop_data>0].mean()+pop_data[pop_data>0].std())
    ax[2].set_title('Population')

    rows, cols = np.where(pop_data>0)
    pop = 0
    dist = 0
    print((wd_data[rows, cols] * pop_data[rows, cols]).sum() / (pop_data[rows, cols].sum()))

