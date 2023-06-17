from pathlib import Path

CATEGORICAL_COLUMNS = ["POLYMER CLASS"]

COUPLED_COLUMNS = [
    ('Molar Volume Vm.1', 'Unnamed: 42', 'MOLAR VOL'),
    ('Density ρ.1', 'Unnamed: 44', 'DENSITY'),
    ('Solubility Parameter δ.1', 'Preferred Solubility Parameter δ', 'SOLUBILITY'),
    ('Molar Cohesive Energy Ecoh.1', 'Preferred Molar Cohesive Energy Ecoh', 'COH ENERGY'),
    ('Glass Transition Temperature Tg.1', 'Unnamed: 50', 'GLASS TEMP'),
    ('Molar Heat Capacity Cp.1', 'Preferred Molar Heat Capacity Cp', 'HEAT CAP'),
    ('Entanglement Molecular Weight Me.1', 'Preferred Entanglement Molecular Weight Me', 'ENT MOL WEIGHT'),
    ('Index of Refraction n.1', 'Unnamed: 56', 'IND REFRACTION'),
    ('Molecular Weight of Repeat unit', 'Molecular Weight of Repeat unit', 'MOL W REP'),
    ('Van-der-Waals Volume VvW', 'Van-der-Waals Volume VvW', 'VDW VOL',)
]

UNNEEDED_COLUMNS = [
    'NAMES AND IDENTIFIERS OF POLYMER',
    'CurlySMILES',
    'Std. InChI',
    'Std. InChIKey',
    'STRUCTURE',
    'IDENTIFIERS OF MONOMER(S)',
    'CAS #.1',
    'STRUCTURE BASED NAME',
    'ACRONYMS',
    'CAS #',
    'IDENTIFIERS OF MONOMER(S)',
    'COMMON NAMES.1',
    'Thermo-Physical Properties: Experimental / Literature Data ',
    'PROPERTY',
    'PROPERTY.1',
    'Thermo-Physical Properties: Calculated Data ',
]

TRAIN_COLUMNS = [
    'POLYMER CLASS',
    'SMILES',
    'MOL W REP',
    'VDW VOL',
    'MOLAR VOL',
    'DENSITY',
    'SOLUBILITY',
    'COH ENERGY',
    'GLASS TEMP',
    'HEAT CAP',
    'ENT MOL WEIGHT',
    'IND REFRACTION'
]

TARGET_COLUMNS = ["DENSITY"]
