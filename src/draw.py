from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io


def smiles_to_svg(smiles, filename, width=300, height=300):
    mol = Chem.MolFromSmiles(smiles)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg_bytes = drawer.GetDrawingText().encode()
    with open(filename, 'wb') as f:
        f.write(svg_bytes)