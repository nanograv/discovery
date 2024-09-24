import json
from typing import Self

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.feather


def read_chain(fileordir: str) -> pd.DataFrame:
    """Read a Discovery Feather chain or a PTMCMC chain directory to a Pandas table.

    This function can read either a Feather file or a PTMCMC chain directory and convert it into a Pandas DataFrame.
    The resulting DataFrame will contain chain data and additional metadata in its `attrs` attribute.

    Parameters
    ----------
    fileordir : str
        Path to either a Feather file or a PTMCMC chain directory.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the chain data. The DataFrame's `attrs` attribute will contain `priors`,
        `runtime_info`, and `noisedict` if available.

    Notes
    -----
    - For Feather files, metadata is stored in the `attrs` attribute of the DataFrame.
    - For PTMCMC directories, the function reads multiple files to construct the DataFrame and its metadata.

    """
    if fileordir.endswith('.feather'):
        table = pyarrow.feather.read_table(fileordir)

        df = table.to_pandas()
        if b'json' in table.schema.metadata:
            df.attrs = json.loads(table.schema.metadata[b'json'].decode('ascii'))

        return df
    else:
        # we'll assume it's a PTMCMC directory
        dirname = fileordir
        pars = list(map(str.strip, open(f'{dirname}/pars.txt', 'r').readlines()))

        df = pd.read_csv(f'{dirname}/chain_1.0.txt', delim_whitespace=True,
                         names=pars + ['logp', 'logl', 'accept', 'pt'])

        for col in df.columns:
            df[col] = df[col].astype(np.float32)

        noisedict = {}
        for line in open(f'{dirname}/runtime_info.txt', 'r'):
            if 'Constant' in line:
                t, v = line.split('=')
                n, c = line.split(':')
                noisedict[n] = float(v)

        df.attrs['priors'] = list(map(str.strip, open(f'{dirname}/priors.txt', 'r').readlines())),
        df.attrs['runtime_info'] = list(map(str.strip, open(f'{dirname}/runtime_info.txt', 'r').readlines())),
        df.attrs['noisedict'] = noisedict

        return df

def save_chain(df: pd.DataFrame, filename: str) -> None:
    """Saves Pandas chain table to Feather, preserving `attrs` in `schema.metadata['json']`.

    This function saves a Pandas DataFrame containing chain data to a Feather file, ensuring that
    the DataFrame's `attrs` are preserved in the Feather file's metadata.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the chain data to be saved.
    filename : str
        The path where the Feather file should be saved.

    Returns
    -------
    None

    Notes
    -----
    The function stores the DataFrame's 'attrs' in the Feather file's schema metadata under the 'json' key.

    """
    table = pyarrow.Table.from_pandas(df)
    table = table.replace_schema_metadata({**table.schema.metadata, 'json': json.dumps(df.attrs)})
    pyarrow.feather.write_feather(table, filename)


class Pulsar:
    """A class representing a pulsar and its associated data.

    This class provides methods to read and write pulsar data from/to Feather files,
    and stores various pulsar attributes such as timing data, residuals, and metadata.

    Attributes
    ----------
    name : str
        The name of the pulsar.
    toas : np.ndarray
        Time of arrivals.
    stoas : np.ndarray
        Site arrival times.
    toaerrs : np.ndarray
        TOA errors.
    residuals : np.ndarray
        Timing residuals.
    freqs : np.ndarray
        Observing frequencies.
    backend_flags : np.ndarray
        Backend flags.
    Mmat : np.ndarray
        Design matrix.
    sunssb : np.ndarray
        Sun's position in the Solar System Barycenter.
    pos_t : np.ndarray
        Pulsar position at each epoch.
    planetssb : np.ndarray
        Planets' positions in the Solar System Barycenter.
    flags : dict
        Various flags associated with the pulsar data.
    dm : float
        Dispersion measure.
    pdist : float
        Pulsar distance.
    pos : list
        Pulsar position.
    phi : float
        Ecliptic longitude.
    theta : float
        Ecliptic latitude.
    noisedict : dict, optional
        Dictionary containing noise model parameters.

    Methods
    -------
    read_feather(filename)
        Class method to read pulsar data from a Feather file.
    save_feather(filename, noisedict=None)
        Instance method to save pulsar data to a Feather file.

    """
    # notes: currently ignores _isort/__isort and gets sorted versions

    columns = ['toas', 'stoas', 'toaerrs', 'residuals', 'freqs', 'backend_flags']
    vector_columns = ['Mmat', 'sunssb', 'pos_t']
    tensor_columns = ['planetssb']
    # flags are done separately

    metadata = ['name', 'dm', 'dm', 'dmx', 'pdist', 'pos', 'phi', 'theta']

    def __init__(self):
        pass

    def __str__(self):
        return f'<Pulsar {self.name}: {len(self.residuals)} res, {self.Mmat.shape[1]} pars>'

    def __repr__(self):
        return str(self)

    @classmethod
    def read_feather(cls, filename: str) -> Self:
        """Read pulsar data from a Feather file.

        This class method reads a Feather file containing pulsar data and returns a new Pulsar object.

        Parameters
        ----------
        filename : str
            Path to the Feather file containing the pulsar data.

        Returns
        -------
        Pulsar
            A new Pulsar object populated with data from the Feather file.

        """
        f = pyarrow.feather.read_table(filename)
        self = Pulsar()

        for array in Pulsar.columns:
            if array in f.column_names:
                setattr(self, array, f[array].to_numpy())

        for array in Pulsar.vector_columns:
            cols = [c for c in f.column_names if c.startswith(array)]
            setattr(self, array, np.array([f[col].to_numpy() for col in cols]).swapaxes(0,1).copy())

        for array in Pulsar.tensor_columns:
            rows = sorted(set(['_'.join(c.split('_')[:-1]) for c in f.column_names if c.startswith(array)]))
            cols = [[c for c in f.column_names if c.startswith(row)] for row in rows]
            setattr(self, array,
                    np.array([[f[col].to_numpy() for col in row] for row in cols]).swapaxes(0,2).swapaxes(1,2).copy())

        self.flags = {}
        for array in [c for c in f.column_names if c.startswith('flags_')]:
            self.flags['_'.join(array.split('_')[1:])] = f[array].to_numpy()

        meta = json.loads(f.schema.metadata[b'json'])
        for attr in Pulsar.metadata:
            setattr(self, attr, meta[attr])
        if 'noisedict' in meta:
            setattr(self, 'noisedict', meta['noisedict'])

        return self

    to_list = lambda a: a.tolist() if isinstance(a, np.ndarray) else a

    def save_feather(self, filename: str, noisedict: dict | None=None) -> None:
        """Save pulsar data to a Feather file.

        This method saves the pulsar data to a Feather file, including all attributes and metadata.

        Parameters
        ----------
        filename : str
            Path where the Feather file should be saved.
        noisedict : dict, optional
            A dictionary of noise model parameters to be saved with the pulsar data.
            If None, the method will use the Pulsar object's noisedict attribute if it exists.

        Returns
        -------
        None

        """
        pydict = {array: getattr(self, array) for array in Pulsar.columns}

        pydict.update({f'{array}_{i}': getattr(self, array)[:,i] for array in Pulsar.vector_columns
                                                                 for i in range(getattr(self, array).shape[1])})

        pydict.update({f'{array}_{i}_{j}': getattr(self, array)[:,i,j] for array in Pulsar.tensor_columns
                                                                 for i in range(getattr(self, array).shape[1])
                                                                 for j in range(getattr(self, array).shape[2])})

        pydict.update({f'flags_{flag}': self.flags[flag] for flag in self.flags})

        meta = {attr: Pulsar.to_list(getattr(self, attr)) for attr in Pulsar.metadata}

        # use attribute if present
        noisedict = getattr(self, 'noisedict', None) if noisedict is None else noisedict
        if noisedict:
            meta['noisedict'] = {par: val for par, val in noisedict.items() if par.startswith(self.name)}

        pyarrow.feather.write_feather(pyarrow.Table.from_pydict(pydict, metadata={'json': json.dumps(meta)}),
                                      filename)
