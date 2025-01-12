import json

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.feather

def read_chain(fileordir):
    """Read a Discovery Feather chain or a PTMCMC chain directory to a Pandas table.
    Look in `attrs` for priors, runtime_info, and noisedict."""

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

def save_chain(df, filename):
    """Saves Pandas chain table to Feather, preserving `attrs` in `schema.metadata['json']`."""

    table = pyarrow.Table.from_pandas(df)
    table = table.replace_schema_metadata({**table.schema.metadata, 'json': json.dumps(df.attrs)})
    pyarrow.feather.write_feather(table, filename)


class Pulsar:
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
    def read_feather(cls, filename):
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
            self.flags['_'.join(array.split('_')[1:])] = f[array].to_numpy().astype('U')

        meta = json.loads(f.schema.metadata[b'json'])

        for attr in Pulsar.metadata:
            if attr in meta:
                setattr(self, attr, meta[attr])
            else:
                print(f'Pulsar.read_feather: cannot find {attr} in feather file {filename}.')

        if 'noisedict' in meta:
            setattr(self, 'noisedict', meta['noisedict'])

        return self

    to_list = lambda a: a.tolist() if isinstance(a, np.ndarray) else a

    def save_feather(self, filename, noisedict=None):
        pydict = {array: getattr(self, array) for array in Pulsar.columns}

        pydict.update({f'{array}_{i}': getattr(self, array)[:,i] for array in Pulsar.vector_columns
                                                                 for i in range(getattr(self, array).shape[1])})

        pydict.update({f'{array}_{i}_{j}': getattr(self, array)[:,i,j] for array in Pulsar.tensor_columns
                                                                 for i in range(getattr(self, array).shape[1])
                                                                 for j in range(getattr(self, array).shape[2])})

        pydict.update({f'flags_{flag}': self.flags[flag] for flag in self.flags})

        meta = {}

        for attr in Pulsar.metadata:
            if hasattr(self, attr):
                meta[attr] = Pulsar.to_list(getattr(self, attr))
            else:
                print(f'Pulsar.save_feather: cannot find {attr} in Pulsar {self.name}.')

        # use attribute if present
        noisedict = getattr(self, 'noisedict', None) if noisedict is None else noisedict
        if noisedict:
            meta['noisedict'] = {par: val for par, val in noisedict.items() if par.startswith(self.name)}

        pyarrow.feather.write_feather(pyarrow.Table.from_pydict(pydict, metadata={'json': json.dumps(meta)}),
                                      filename)
