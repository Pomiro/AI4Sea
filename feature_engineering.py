import os
import numpy as np
import pandas as pd
import pathlib
import pickle
from geopy.distance import geodesic
from geopy.distance import great_circle as GRC
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# closest ports locations
PORTS_LOCATIONS = {'Vanino': (49.087249402089476, 140.27050021926925),
                   'Ohotsk': (60.02300956412593, 143.50221973254563),
                   'Amur': (55.33307577484239, 141.03909589529843),
                   'Kastri': (51.50009866145172, 140.83986910388506),
                   'Magadan': (59.64915073230776, 150.7250524094824),
                   'Severo_Kurils': (50.679979, 156.141158),
                   'Ribnaya_baza_4': (56.341076, 155.829650)}

# kmeans clustering by ports distance model weighs filename
KMEANS_PORTS_WEIGHTS = "kmeans_ports.pkl"
# kmeans clustering by coordinates model weighs filename
KMEANS_COORD_WEIGHTS = "kmeans_geo.pkl"


class FeaturePipeline:
    """Creates several feature on given pollock data"""

    def __init__(self, logger, models_weights):
        self.input_data = None
        self.ports_col_names = []
        self.kmeans_geo = None
        self.kmeans_ports = None
        self.logger = logger
        self.models_weights = models_weights

    def _geo_dist(self, loc1, loc2):
        """
        Calculate distances between lat/long points
        """
        try:
            dist = geodesic(loc1, loc2).km
        except ValueError:
            dist = 0
        return dist

    def _ships_route_filter(self, target_column):
        """
        Filtering by ships and their routes
        """
        return self.input_data.sort_values(['idves', 'datetimes']).groupby(['idves', 'date'])[target_column]

    def _fill_val(self):
        """
        Dummy filler for missing data
        """
        self.logger.info("Add features: fill missing values")
        # add date atr to simpler filtering
        self.input_data['date'] = self.input_data['datetimes'].dt.date
        # fill velocity
        self.input_data['velocity'] = self._ships_route_filter('velocity').apply(
            lambda group: group.interpolate(method='nearest') if np.count_nonzero(np.isnan(group)) < (
                        len(group) - 1) else group)
        self.input_data['velocity'].fillna(0, inplace=True)
        # fill course
        self.input_data['course'] = self._ships_route_filter('course').apply(
            lambda group: group.interpolate(method='nearest') if np.count_nonzero(np.isnan(group)) < (
                        len(group) - 1) else group)
        self.input_data['course'] = self.input_data['course'].fillna(0).astype(float)

        self.input_data['total_total_ton'] = self._ships_route_filter('total_ton').cumsum()
        self.input_data['total_ton_median'] = self._ships_route_filter('total_ton').transform('median')
        self.input_data['total_ton_diff'] = self._ships_route_filter('total_ton').diff()



    def _add_route_dependencies(self):
        """
        Add several route dependencies as previous route points data, total characteristics etc.
        """
        self.logger.info("Add route dependencies: create total points per route value")
        self.input_data['total_route_points'] = self._ships_route_filter('idinf').transform("count")
        self.input_data['total_route_points'] = self.input_data['total_route_points'].astype(float)

        # add distances to closest ports
        self.logger.info("Add route dependencies: calculate distances from nearest ports to ship location")
        self.input_data['lat_long'] = tuple(zip(self.input_data.latitude, self.input_data.longitude))

        for name, loc in PORTS_LOCATIONS.items():
            col_name = f'dist_to_port_{name}'
            if col_name not in self.ports_col_names:
                self.ports_col_names.append(col_name)
            self.input_data[col_name] = self.input_data['lat_long'].apply(lambda x: self._geo_dist(x, loc))
        # scale lat/long values
        self.input_data['latitude'] = self.input_data['latitude'] * 10000
        self.input_data['longitude'] = self.input_data['longitude'] * 10000

        self.logger.info("Add route dependencies: calculate time passed from previous route point")
        self.input_data['time_from_prev_record'] = self._ships_route_filter('datetimes').diff()
        self.input_data['time_from_prev_record'] = pd.to_timedelta(
            self.input_data['time_from_prev_record']) / pd.Timedelta('60s')
        self.input_data['time_from_prev_record'].fillna(0, inplace=True)


        self.logger.info("Add route dependencies: calculate total time spent")
        self.input_data['total_time'] = self._ships_route_filter('time_from_prev_record').cumsum()

        self.logger.info("Add route dependencies: calculate distance passed from previous route point")
        self.input_data['dist_from_prev_record'] = self._ships_route_filter('lat_long').shift()
        self.input_data['dist_from_prev_record'] = self.input_data.apply(
            lambda x: self._geo_dist(x['dist_from_prev_record'], x['lat_long']),
            axis=1)

        self.logger.info("Add route dependencies: calculate total dist passed")
        self.input_data['total_dist'] = self._ships_route_filter('dist_from_prev_record').cumsum()

        self.logger.info("Add route dependencies: calculate mean velocity based on previous record")
        self.input_data['mean_velocity'] = self.input_data['dist_from_prev_record'] / (
                self.input_data['time_from_prev_record'] / 60)
        # fill  missing values
        self.input_data.loc[self.input_data['mean_velocity'] == np.inf, ['mean_velocity']] = self.input_data['velocity']
        self.input_data.loc[self.input_data['mean_velocity'].isnull(), ['mean_velocity']] = self.input_data['velocity']

        # scale values to avoid overfitting
        scale_cols = ['total_dist',
                      'dist_from_prev_record', 
                      'velocity', 
                      'mean_velocity',
                      ]
        for col_name in scale_cols:
            self.input_data[col_name] = self.input_data.sort_values(['idves', 'datetimes']).groupby(
                ['idves', 'date']).apply(
                lambda x: pd.DataFrame(MinMaxScaler().fit_transform(x[[col_name]]),
                                       index=x.index))
        # add previous route point values
        prev_cols = ['latitude', 'longitude', 'velocity', 'mean_velocity']
        self.logger.info(f"Add route dependencies: add {prev_cols} data from previous route point")
        for col_name in prev_cols:
            self.input_data[f'previous_{col_name}'] = self._ships_route_filter(col_name).shift()
        self.input_data.fillna(0, inplace=True)
        self.input_data.drop(['idves', 'idinf', 'lat_long'], axis=1, inplace=True)

    def _parse_dates(self):
        """
        Parse datetime values to cat features
        """
        self.logger.info(f"Parse dates: add features from dates")
        self.input_data['year'] = self.input_data['datetimes'].dt.year
        self.input_data['month'] = self.input_data['datetimes'].dt.month
        self.input_data['month_sin'] = np.sin(self.input_data['month'] * (2 * np.pi / 12))
        self.input_data['day'] = self.input_data['datetimes'].dt.weekday
        self.input_data['hour'] = self.input_data['datetimes'].dt.hour
        self.input_data['hour_sin'] = np.sin(self.input_data['hour'] * (2 * np.pi / 24))
        self.input_data.drop(['datetimes', 'date', 'course','hour', 'month'], axis=1, inplace=True)

    def _fit_clusters(self, km_geo_num=7, km_ports_num=7):
        """
        Fitting KMeans clusters for ships coordinates and ports distances data
        """
        self.logger.info(f"Clustering: fitting clusters")
        clusters_data = self.input_data.loc[
            self.input_data['science'] == 0, ['latitude', 'longitude'] + self.ports_col_names]

        self.logger.info(f"Clustering: fitting coords cluster based on ships locations")
        self.kmeans_geo = KMeans(n_clusters=km_geo_num, init='k-means++')
        self.kmeans_geo.fit(clusters_data[clusters_data.columns[0:2]])

        with open(os.path.join(self.models_weights, KMEANS_COORD_WEIGHTS), "wb") as f:
            pickle.dump(self.kmeans_geo, f)

        self.logger.info(f"Clustering: fitting ports cluster based on distances to ports")
        self.kmeans_ports = KMeans(n_clusters=km_ports_num, init='k-means++')
        self.kmeans_ports.fit(clusters_data[clusters_data.columns[2:8]])

        with open(os.path.join(self.models_weights, KMEANS_PORTS_WEIGHTS), "wb") as f:
            pickle.dump(self.kmeans_ports, f)

    def _add_clusters(self):
        """
        Predicting values (coord and ports distances cluster) based on fitted objects
        """
        if None in (self.kmeans_geo, self.kmeans_ports):
            try:
                with open(os.path.join(self.models_weights, KMEANS_COORD_WEIGHTS), "rb") as f:
                    self.kmeans_geo = pickle.load(f)
                with open(os.path.join(self.models_weights, KMEANS_PORTS_WEIGHTS), "rb") as f:
                    self.kmeans_ports = pickle.load(f)
            except:
                raise ValueError('kmeans objects not fitted/saved, firstly train data should be set used')

        clusters_data = self.input_data.loc[:, ['latitude', 'longitude'] + self.ports_col_names]
        self.logger.info(f"Clustering: predicting values with fitted coord_cluster")
        self.input_data['coord_cluster'] = self.kmeans_geo.predict(clusters_data[clusters_data.columns[0:2]])

        self.logger.info(f"Clustering: predicting values with fitted port_cluster")
        self.input_data['port_cluster'] = self.kmeans_ports.predict(clusters_data[clusters_data.columns[2:8]])

    def make_features(self,
                      input_data,
                      path_to_save='',
                      train=True):
        """
        Run feature pipeline, save output data
        path_to_save: (str) path to save data after feature generation
        input_data: (pandas.core.frame.DataFrame) input data to generate features
        train: (bool) flag that describes train or dev data would be used
        """
        self.logger.info(f"Starting data pipeline on {'train' if train else 'val/test'} data")
        self.input_data = input_data.copy()
        self._fill_val()
        self._add_route_dependencies()
        self._parse_dates()

        if train:
            self._fit_clusters()

        self._add_clusters()
        self.logger.info(f"Save data with features")
        if path_to_save:
            self.input_data.to_csv(path_to_save)
        return self.input_data
