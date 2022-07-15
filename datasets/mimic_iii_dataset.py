"""
Data loaders for MIMIC-III

NOTE: raw data can be downloaded from https://physionet.org/content/mimiciii/1.4/. You will need to first run
the code from https://github.com/USC-Melady/Benchmarking_DL_MIMICIII and then the Jupyter notebook at
notebooks/Tabular_Mimic-III_Preprocessing.ipynb
"""
import os

import pandas as pd

from aif360.datasets import StandardDataset


def default_preprocessing(df):
    """Remove missing values for the dataframe."""
    df.loc[(df['marital_status'] != 'MARRIED') & (df['marital_status'] != 'SINGLE'), 'marital_status'] = 1
    df.loc[(df['marital_status'] == 'MARRIED') | (df['marital_status'] == 'SINGLE'), 'marital_status'] = 0
    df.loc[df['gender'] == 'M', 'gender'] = 0
    df.loc[df['gender'] == 'F', 'gender'] = 1
    df.loc[(df['insurance'] != 'Medicare') & (df['insurance'] != 'Medicaid'), 'insurance'] = 1
    df.loc[(df['insurance'] == 'Medicare') | (df['insurance'] == 'Medicaid'), 'insurance'] = 0
    df.loc[(df['ethnicity'] != 'WHITE') & (df['ethnicity'] != 'WHITE - RUSSIAN') &
           (df['ethnicity'] != 'WHITE - OTHER EUROPEAN') & (df['ethnicity'] != 'WHITE - BRAZILIAN') &
           (df['ethnicity'] != 'WHITE - EASTERN EUROPEAN'), 'ethnicity'] = 1
    df.loc[(df['ethnicity'] == 'WHITE') | (df['ethnicity'] == 'WHITE - RUSSIAN') |
           (df['ethnicity'] == 'WHITE - OTHER EUROPEAN') | (df['ethnicity'] == 'WHITE - BRAZILIAN') |
           (df['ethnicity'] == 'WHITE - EASTERN EUROPEAN'), 'ethnicity'] = 0
    return df


class MimicDataset(StandardDataset):
    """MIMIC Dataset.

    See :file:`aif360/data/raw/mimic-iii/README.md`.
    """

    def __init__(self, label_name='hospital_mort', favorable_classes=[1],
                 protected_attribute_names=['insurance'],
                 privileged_classes=[[0]],
                 instance_weights_name=None,
                 categorical_features=['mingcs','aids','hem','mets','admissiontype'],
                 features_to_keep=[],
                 features_to_drop=['1_day_mort','2_day_mort','3_day_mort','30_day_mort','1_year_mort',
                                   'religion','gender','marital_status','ethnicity'],
                 na_values='', custom_preprocessing=default_preprocessing,
                 metadata=None):
        # See :obj:`StandardDataset` for a description of the arguments.

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'mimic-iii', 'mimic_non_series.csv')

        try:
            df = pd.read_csv(filepath, na_values=na_values)
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\nunzip it and place the files, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'mimic-iii'))))
            import sys
            sys.exit(1)

        super(MimicDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)
