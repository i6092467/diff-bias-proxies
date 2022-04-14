"""
General tabular dataset structure, a wrapper for the AIF 360 toolkit.
"""
import numpy as np


def get_data(dataset, protected_attribute, seed=101, config=None):
    def protected_attribute_error():
        raise ValueError(f'protected attribute {protected_attribute} is not available for dataset {dataset}')

    if dataset == 'adult':
        from aif360.datasets import AdultDataset
        dataset_orig = AdultDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        elif protected_attribute == 'sex_or_race':
            dataset_orig.feature_names += ['sex_or_race']
            dataset_orig.features = np.hstack([dataset_orig.features, np.expand_dims(np.logical_or(*dataset_orig.features[:, [2, 3]].T).astype(np.float64), -1)])
            dataset_orig.protected_attributes = np.hstack([dataset_orig.protected_attributes, dataset_orig.features[:, [-1]]])
            dataset_orig.protected_attribute_names += ['sex_or_race']
            dataset_orig.privileged_protected_attributes += [np.array([1.])]
            dataset_orig.unprivileged_protected_attributes += [np.array([0.])]
            privileged_groups = [{'sex_or_race': 1}]
            unprivileged_groups = [{'sex_or_race': 0}]
        elif protected_attribute == 'race':
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'german':
        from aif360.datasets import GermanDataset
        dataset_orig = GermanDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 1}]
            unprivileged_groups = [{'sex': 0}]
        elif protected_attribute == 'age':
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'compas':
        from aif360.datasets import CompasDataset
        dataset_orig = CompasDataset()
        if protected_attribute == 'sex':
            privileged_groups = [{'sex': 0}]
            unprivileged_groups = [{'sex': 1}]
        elif protected_attribute == 'sex_or_race':
            dataset_orig.feature_names += ['sex_or_race']
            dataset_orig.features = np.hstack([dataset_orig.features, np.expand_dims(np.logical_or(*dataset_orig.features[:, [0, 2]].T).astype(np.float64), -1)])
            dataset_orig.protected_attributes = np.hstack([dataset_orig.protected_attributes, dataset_orig.features[:, [-1]]])
            dataset_orig.protected_attribute_names += ['sex_or_race']
            dataset_orig.privileged_protected_attributes += [np.array([1.])]
            dataset_orig.unprivileged_protected_attributes += [np.array([0.])]
            privileged_groups = [{'sex_or_race': 1}]
            unprivileged_groups = [{'sex_or_race': 0}]
        elif protected_attribute == 'race':
            privileged_groups = [{'race': 1}]
            unprivileged_groups = [{'race': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'bank':
        from aif360.datasets import BankDataset
        dataset_orig = BankDataset()
        if protected_attribute == 'age':
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'diabetes':
        from aif360.datasets import DiabetesDataset
        dataset_orig = DiabetesDataset()
        if protected_attribute == 'race':
            privileged_groups = [{'race': 0}]
            unprivileged_groups = [{'race': 1}]
        else:
            protected_attribute_error()

    elif dataset == 'mimic':
        from datasets.mimic_iii_dataset import MimicDataset
        dataset_orig = MimicDataset()
        if protected_attribute == 'age':
            privileged_groups = [{'age': 0}]
            unprivileged_groups = [{'age': 1}]
        elif protected_attribute == 'marital_status':
            privileged_groups = [{'marital_status': 0}]
            unprivileged_groups = [{'marital_status': 1}]
        elif protected_attribute == 'gender':
            privileged_groups = [{'gender': 0}]
            unprivileged_groups = [{'gender': 1}]
        elif protected_attribute == 'insurance':
            privileged_groups = [{'insurance': 0}]
            unprivileged_groups = [{'insurance': 1}]
        elif protected_attribute == 'ethnicity':
            privileged_groups = [{'ethnicity': 0}]
            unprivileged_groups = [{'ethnicity': 1}]
        else:
            protected_attribute_error()

    elif dataset == 'synthetic_loh':
        from datasets.simulations import SyntheticLohDataset
        if config is None:
            dataset_orig = SyntheticLohDataset(seed=seed)
        else:
            dataset_orig = SyntheticLohDataset(n=config['dataset_size'], alpha=config['dataset_alpha'], seed=seed)
        if protected_attribute == 'Z':
            privileged_groups = [{'Z': 1}]
            unprivileged_groups = [{'Z': 0}]
        else:
            protected_attribute_error()

    elif dataset == 'synthetic_zafar':
        from datasets.simulations import SyntheticZafarDataset
        if config is None:
            dataset_orig = SyntheticZafarDataset(seed=seed)
        else:
            dataset_orig = SyntheticZafarDataset(n=config['dataset_size'], theta=config['dataset_theta'], seed=seed)
        if protected_attribute == 'Z':
            privileged_groups = [{'Z': 1}]
            unprivileged_groups = [{'Z': 0}]
        else:
            protected_attribute_error()

    else:
        raise ValueError(f'{dataset} is not an available dataset.')

    dataset_orig_train, dataset_orig_vt = dataset_orig.split([0.6], shuffle=True, seed=seed)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True, seed=seed)

    return dataset_orig_train, dataset_orig_valid, dataset_orig_test, privileged_groups, unprivileged_groups
