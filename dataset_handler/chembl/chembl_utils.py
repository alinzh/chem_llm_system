import pandas as pd


class ChemblLoader:
    """
    A class to load and filter data from a ChEMBL dataset.

    Attributes:
        columns (dict): A dictionary mapping column names to their data types.
        data_path (str): The path to the CSV file.
        data (pd.DataFrame or None): The loaded data, or None if not loaded.
    """

    def __init__(self, load: bool = False, file_path: str = None):
        """
        Initializes the ChemblLoader with optional data loading.

        Args:
            load (bool): If True, the data will be loaded immediately upon initialization.
            file_path (str): The path to the CSV file containing the ChEMBL data.
        """
        self.columns = chembl_columns = {
            'ChEMBL ID': str,
            'Name': str,
            'Synonyms': str,
            'Type': str,
            'Max Phase': int,
            'Molecular Weight': float,
            'Targets': int,
            'Bioactivities': int,
            'AlogP': float,
            'Polar Surface Area': float,
            'HBA': int,
            'HBD': int,
            '#RO5 Violations': int,
            '#Rotatable Bonds': int,
            'Passes Ro3': object,  
            'QED Weighted': float,
            'CX Acidic pKa': float,
            'CX Basic pKa': float,  
            'CX LogP': float,
            'CX LogD': float,
            'Aromatic Rings': int,
            'Structure Type': str,
            'Inorganic Flag': int,
            'Heavy Atoms': int,
            'HBA (Lipinski)': int,
            'HBD (Lipinski)': int,
            '#RO5 Violations (Lipinski)': int,
            'Molecular Weight (Monoisotopic)': float,
            'Np Likeness Score': float,
            'Molecular Species': str,
            'Molecular Formula': str,
            'Smiles': str,
            'Inchi Key': str,
            'Inchi': str,
            'Withdrawn Flag': bool,
            'Orphan': int,
            'Records Key': object, 
            'Records Name': object  
        }
        self.data_path = file_path
        if load and file_path:
            self.data = self._load_data()
        else:
            self.data = None
            
    def _load_data(self):
        """
        Loads the data from the CSV file specified by data_path.

        Returns:
            None
        """
        self.data = pd.read_csv(self.data_path, delimiter=";", on_bad_lines="skip", engine="python")
        
    def get_columns(self) -> dict:
        """
        Returns the columns dictionary.

        Returns:
            dict: The columns dictionary.
        """
        return self.columns
    
    def get_filtered_data(self, selected_columns: list, filters: dict = None) -> pd.DataFrame:
        """
        Filters the data based on the selected columns and filter conditions.

        Args:
            selected_columns (list): A list of columns to include in the filtered data.
            filters (dict, optional): A dictionary of filter conditions. The keys are column names, and the values are conditions. 
                                      Conditions can be:
                                      - A tuple (min_val, max_val) for range filtering.
                                      - A string or boolean for exact matching.

        Returns:
            pd.DataFrame: The filtered DataFrame.
        """
        if 'Smiles' not in selected_columns:
            selected_columns.append('Smiles')
            
        if not self.data:
            self._load_data()
        df = self.data[selected_columns]
    
        if filters:
            for column, condition in filters.items():
                if column in df.columns:
                    if isinstance(condition, tuple) and len(condition) == 2:
                        min_val, max_val = condition
                        df = df[(df[column] >= min_val) & (df[column] <= max_val)]
                    elif isinstance(condition, (str, bool)):
                        df = df[df[column] == condition]
        
        return df
    
if __name__ == "__main__":
    file_path = '/Users/alina/Desktop/ИТМО/chem_llm_system/dataset_handler/chembl/p1.csv'
    selected_columns = ["Molecular Weight"]  
    filters = {
        "Molecular Weight": (150, 500), 
    }
    client = ChemblLoader(True, file_path)
    df = client.get_filtered_data(selected_columns, filters)
    print(df)