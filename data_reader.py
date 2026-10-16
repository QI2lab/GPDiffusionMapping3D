import numpy as np
import copy
from types import SimpleNamespace

# CAPITALIZED COMMENTS FOR CLARITY

# THIS IS THE OBJECT OF DATA
DATA = {
    # NEED TO BE LOADED IN
    'trajectories': None,       # Coordinates of the data
    'trajectoriesIndex': None,  # Index of trajectory number
    'deltaT': None,             # Time passed in between each frame of data

    # INITIALIZED INDEPENDENTLY BASED ON PARAMETERS ABOVE
    'nData': None,              # Number of data points (set to len(nData))
    'nTrajectories': None,      # Number of trajectories (set to len(np.unique(data.trajectoriesIndex)))
}

def data_reader(path, scale=1, deltaT = 1/30):
    """
    Read data from a CSV file and prepare it for further processing.

    Args:
        path (str): Path to the CSV file.
        scale (float): Scaling factor to transform data to nm. Defaults to 1.
        deltaT (float): Time interval between each frame of data. Defaults to 1/30.

    Returns:
        types.SimpleNamespace: An object containing trajectory data and related information.
    """

    # Read the CSV file, considering the header
    data = np.genfromtxt(path, delimiter=', ', skip_header=1)

    # Separate columns into individual arrays
    dataVectIndex = data[:, 0]
    dataVect = data[:, 1:]
    
    # Localization adjustment to nanometers
    dataVect = dataVect[::]*scale
    dataVectIndex = dataVectIndex[::]
    
    # Put time step manually as unavailable from data file
    deltaT = deltaT

    # Return dataVect, dataVectIndex, deltaT
    data = SimpleNamespace(**copy.deepcopy(DATA))
    data.trajectoriesIndex = dataVectIndex
    data.trajectories = dataVect
    data.deltaT = deltaT
    data.nData = len(data.trajectoriesIndex)
    data.nTrajectories = len(np.unique(data.trajectoriesIndex))

    return data

def prepare_tracks(particles, positions, timestamps, scale=1):
    """Prepare sorted within-particle 2D or ZYX 3D transitions.

    Parameters
    ----------
    particles : array-like
        Already linked identities, retained without floating-point conversion.
    positions : array-like
        XY columns for 2D or ZYX columns for 3D, in input length units.
    timestamps : array-like
        Numeric observation times in seconds.
    scale : float
        Positive conversion from input length units to nanometers.

    Returns
    -------
    types.SimpleNamespace
        Sorted tracks, consecutive start/end positions and positive elapsed times.
    """
    coordinates=np.asarray(positions,dtype=np.float64)
    times=np.asarray(timestamps,dtype=np.float64)
    identities=np.asarray(particles,dtype=object)
    if coordinates.ndim!=2 or coordinates.shape[1] not in (2,3):
        raise ValueError("Positions must have two XY or three ZYX columns")
    if times.ndim!=1 or identities.ndim!=1 or len(times)!=len(coordinates) or len(identities)!=len(times):
        raise ValueError("Particle, position and timestamp rows must agree")
    if not np.isfinite(scale) or scale<=0:
        raise ValueError("scale must be finite and positive")
    coordinates=coordinates*scale
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(times)):
        raise ValueError("Positions and numeric timestamps must be finite")
    groups={}
    for i,identity in enumerate(identities):
        if identity is None or (isinstance(identity, str) and not identity) or (isinstance(identity,(float,np.floating)) and not np.isfinite(identity)):
            raise ValueError("Particle identities must not be missing")
        groups.setdefault(identity,[]).append(i)
    order=[]; starts=[]; ends=[]; intervals=[]
    for indices in groups.values():
        indices=np.asarray(indices,dtype=np.int64)
        indices=indices[np.argsort(times[indices],kind='stable')]
        tau=np.diff(times[indices])
        if not np.all(np.isfinite(tau)) or np.any(tau<=0):
            raise ValueError("Each particle must have strictly increasing distinct timestamps")
        order.extend(indices)
        if len(indices)>1:
            starts.append(coordinates[indices[:-1]])
            ends.append(coordinates[indices[1:]])
            intervals.append(tau)
    if not intervals:
        raise ValueError("Tracks must contain at least one positive-time transition")
    return SimpleNamespace(trajectories=coordinates[order],trajectoriesIndex=identities[order],
        timestamps=times[order],nData=len(times),nTrajectories=len(groups),
        dimensions=coordinates.shape[1],starts=np.concatenate(starts),ends=np.concatenate(ends),
        delta_t=np.concatenate(intervals))


def data_reader_3d(path, scale=1):
    """Read already-linked particle #, z, y, x, timestamp CSV observations."""
    import csv
    from pathlib import Path

    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream)
        required=('particle #','z','y','x','timestamp')
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"3D CSV requires columns {required}")
        rows=list(reader)
    try:
        return prepare_tracks([row['particle #'] for row in rows],
            [[float(row[c]) for c in ('z','y','x')] for row in rows],
            [float(row['timestamp']) for row in rows],scale=scale)
    except (TypeError,ValueError) as error:
        raise ValueError(f"Invalid timestamped 3D tracks: {error}") from error
