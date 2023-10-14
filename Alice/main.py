import numpy as np
import csv
import tensorflow as tf

"""
Alice is the main project of a non-feedforward probabilistic model
Alice is to collect up to 1000 data streams and seemingly unrelated data in nature.
Alice will develop learn the data from real-time, then simulate her own version of the system she has been exposed to.

Client -> Listen from server.
Server -> Sensors
Input Datastream -> continuous update -> data format: (time, sensor_output, location, [direction if camera data])
Output Datastream -> Unknown, Alice will figure it out.

Arch:
    Input -> Client
    Client -> GraphDatabase
    GraphDatabase -> Update weights between datapoints
    GraphDatabase -> K-Means Clustering using n_steams
    Perform Wasserstein loss by cross-checking to each other clusters. (Distribution Difference)
    
    If Wasserstein loss is between -20.0 and 20.0,
        Set possible-anomaly-flag to true
    If Wasserstein loss is outside the above range,
        increase clusters by 1.
    
    Save new CSV file every 1 hour.
    Save row format: (time, sensor, sensor_output, location)
    
    A Siamese network will be used to look for similar features.
    Data points without similar features will be marked as a new cluster
"""