import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from dataset_paths import feature_file, pseudo_file


class FeatureDataset(data.Dataset):
    """
    Args:
        data_dir (String): data file
        labels_dir (String): labels file
        att_dir (String): attributes file
        clu_dir (string): clu lbls file
    """

    def __init__(self, data_dir, labels_dir, att_dir, clu_dir):
        self.features = pd.read_csv(data_dir, header=None, index_col=None).values  # Pre-trained ResNet-50 features of data
        self.labels = pd.read_csv(labels_dir, header=None, index_col=None)[0].values
        self.attributes = pd.read_csv(att_dir, header=None).values
        if len(self.attributes) != len(self.features):
            if self.attributes.shape[0] == 50:
                self.attributes = self.attributes[self.labels.astype(np.int64)]
            else:
                raise ValueError(
                    f"attributes rows {len(self.attributes)} != features {len(self.features)}: {att_dir}"
                )
        self.clu_labels = pd.read_csv(clu_dir, header=None, index_col=None)[0].values  # Clustering results/labels based on the ResNet-50 features of the target domain data
        self.classes, self.counts = np.unique(self.labels, return_counts=True)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, attribute) where target is class_index of the target class.
        """
        feats = self.features[index, :]
        lbls = self.labels[index]
        atts = self.attributes[index, :]
        clu_lbls = self.clu_labels[index]

        return feats, lbls, atts, clu_lbls

    def __len__(self):
        return len(self.labels)


def generate_dataloader(args, pseudo_bank=None):
    src_feat_dir = feature_file(args, 'src', 'feats')
    src_lbl_dir = feature_file(args, 'src', 'labels')
    src_att_dir = feature_file(args, 'src', 'att')
    src_clu_dir = src_lbl_dir

    source_train_dataset = FeatureDataset(
        data_dir=src_feat_dir, labels_dir=src_lbl_dir, att_dir=src_att_dir, clu_dir=src_clu_dir)

    tgt_feat_dir = feature_file(args, 'tgt', 'feats')
    tgt_lbl_dir = feature_file(args, 'tgt', 'labels')
    tgt_att_dir = feature_file(args, 'tgt', 'att')
    tgt_clu_dir = getattr(args, 'tgt_clu_override', '') or pseudo_file(args)

    target_train_dataset = FeatureDataset(
        data_dir=tgt_feat_dir, labels_dir=tgt_lbl_dir, att_dir=tgt_att_dir, clu_dir=tgt_clu_dir)
    target_test_dataset = FeatureDataset(
        data_dir=tgt_feat_dir, labels_dir=tgt_lbl_dir, att_dir=tgt_att_dir, clu_dir=tgt_clu_dir)
    if pseudo_bank is not None:
        pseudo_bank.attach_dataset(target_train_dataset)
        pseudo_bank.attach_dataset(target_test_dataset)

    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    return source_train_loader, target_train_loader, target_test_loader
