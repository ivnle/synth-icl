"""Generator of Omniglot data sequences."""

import logging
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import torchvision
import transformers

import einops
from einops import rearrange, reduce

IGNORE_IDX = -1
IMAGE_SIZE = 105
N_CHARACTER_CLASSES = 1623
N_EXEMPLARS_PER_CLASS = 20


class OmniglotDatasetForSampling:
    """Class for loading Omniglot dataset, used downstream for sampling sequences."""

    def __init__(
        self, omniglot_split, exemplars="single", augment_images=False, n_to_keep=None
    ):
        """Load Omniglot data into memory.

        Args:
          omniglot_split: which Omniglot split to load from ('train'/'test'/'all')
          exemplars: see _load_omniglot_data below.
          augment_images: whether to augment the image classes by also including
            image transforms on the original Omniglot images.
          n_to_keep: Only keep a subset of the Omniglot split.
        """

        # Load the data into memory.
        self.data = self._load_omniglot_data(omniglot_split, exemplars)
        if n_to_keep:
            self.data = {k: self.data[k] for k in range(n_to_keep)}
        if augment_images:
            self.data = self._apply_image_augmentations(self.data)

        self.example_type = "omniglot"
        self.omniglot_split = omniglot_split

    def _load_omniglot_data(self, split, exemplars):
        """Load the Omniglot data into memory.

        Args:
          split: Which Omniglot split to load from ('train'/'test'/'all')
          exemplars: How to handle the 20 exemplars per Omniglot class.
            'single': Only load the first exemplar from each character class.
            'separated': Load all 20 exemplars for each character class, and assign
              them each their own unique class label.
            'all': Load all 20 exemplars for each character class, keeping all 20
              assigned to the same class label (the standard Omniglot problem).

        Returns:
          data: a dict of entries {label: image}
          classes: a sorted list of all labels
        """
        if split == "all":
            data_train = self._load_omniglot_data("train", exemplars)
            data_test = self._load_omniglot_data("test", exemplars)

            # offset the data_test labels with the number of data_train labels
            data_test = {k + len(data_train): v for k, v in data_test.items()}

            data = {**data_train, **data_test}

            logging.info(f"# {split} images: {sum([len(v) for v in data.values()])}")
            logging.info(f"# {split} classes: {len(data.keys())}")
            logging.info(f"example shape: {data[0][0].shape}")

            return data

        elif split == "train":
            ds = torchvision.datasets.Omniglot(
                root="data",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                background=True,
            )
        elif split == "test":
            ds = torchvision.datasets.Omniglot(
                root="data",
                download=True,
                transform=torchvision.transforms.ToTensor(),
                background=False,
            )
        else:
            raise ValueError(
                f"Invalid split {split}. Must be one of 'train', 'test', or 'all'."
            )

        logging.info(f"# {split} images: {len(ds)}")

        data = {}

        for image, label in ds:
            image = image.numpy()
            # reshape from [1, 105, 105] to [105, 105, 1] to make ported tensorflow code happy
            image = np.reshape(image, (105, 105, 1))
            # label = label.numpy().astype(np.int32)
            if exemplars == "single":
                # Populate the dictionary of {label: image} entries.
                # Only add to the dataset if that class doesn't already exist.
                if label not in data:
                    # image = image.numpy()
                    data[label] = image
            else:
                # Populate a dictionary of {label: [images]} entries.
                # Add all images corresponding to a given label.
                if label not in data:
                    data[label] = []
                # image = image.numpy()
                data[label].append(image)

        # If loading exemplars 'separated', re-assign each to a unique label.
        if exemplars == "separated":
            data_orig = data
            data = {}
            for label_orig, images in data_orig.items():
                for i, image in enumerate(images):
                    label_new = label_orig * N_EXEMPLARS_PER_CLASS + i
                    data[label_new] = image

        logging.info(f"# {split} classes: {len(data)}")

        return data

    def _apply_image_augmentations(self, data_orig):
        """Apply transformations to the images to obtain a larger number of classes."""
        i = 0
        data_augmented = {}

        for image in data_orig.values():
            for flip_lr in [False, True]:
                for rot90 in range(4):
                    # Apply the transformations.
                    transformed = image.copy()
                    if flip_lr:
                        transformed = tf.image.flip_left_right(transformed)
                    transformed = tf.image.rot90(transformed, k=rot90)

                    # Convert back into list in the batch dimension.
                    if not isinstance(transformed, list):
                        transformed = [
                            transformed[i] for i in range(transformed.shape[0])
                        ]

                    data_augmented[i] = transformed
                    i += 1

        return data_augmented


class SymbolicDatasetForSampling:
    """Class for loading symbolic (integers) dataset, used downstream for sampling sequences."""

    def __init__(self, dataset_size):
        """Load symbolic (integers) data into memory.

        Args:
          dataset_size: number of integers in the dataset
        """
        # Load the data into memory.
        self.data = {i: i for i in range(dataset_size)}
        self.example_type = "symbolic"


class SeqGenerator:
    """Generates sequences of 'common', 'rare', or Zipf-distributed classes."""

    def __init__(
        self,
        dataset_for_sampling,
        n_rare_classes,
        n_common_classes,
        n_holdout_classes=0,
        zipf_exponent=1.0,
        use_zipf_for_common_rare=False,
        noise_scale=0.1,
        preserve_ordering_every_n=None,
        random_seed=1337,
    ):
        """Split classes into 'common' and 'rare'.

        Args:
          dataset_for_sampling: e.g. OmniglotDatasetForSampling
          n_rare_classes: number of rare classes.
          n_common_classes: number of common classes.
          n_holdout_classes: number of holdout classes
          zipf_exponent: exponent on Zipfian distribution that can be defined over
            combined rare+common classes.
          use_zipf_for_common_rare: if True, common and rare classes will be sampled
            according to the Zipfian distribution that is defined over combined
            rare+common classes. Otherwise, they will be sampled uniformly.
          noise_scale: scale for the Gaussian noise that will be added to each image
          preserve_ordering_every_n: [optional] if provided, the ordering will not
            be shuffled within every n examples. This is useful with if e.g.
            exemplars='separated' or augment_images=True for the Omniglot dataset,
            and we would like to ensure that exemplars derived from the same class
            do not occur in both train and holdout sets.
          random_seed: seed for random generator.
        """
        self.example_type = dataset_for_sampling.example_type
        self.data = dataset_for_sampling.data
        self.classes = sorted(self.data.keys())
        n_classes_orig = len(self.classes)
        logging.info(
            'Loaded %d classes of type "%s"', n_classes_orig, self.example_type
        )

        # Determine which classes belongs to the "rare" vs "common" categories.
        # Set a fixed randomized ordering for rare vs common assignment, to ensure
        # alignment across training and evals.
        rng = np.random.default_rng(random_seed)
        if preserve_ordering_every_n:
            assert n_classes_orig % preserve_ordering_every_n == 0
            n_subgroups = int(n_classes_orig / preserve_ordering_every_n)
            subgroup_ordering = rng.choice(
                range(n_subgroups), size=n_subgroups, replace=False
            )
            class_ordering = np.split(np.arange(n_classes_orig), n_subgroups)
            class_ordering = np.array(class_ordering)[subgroup_ordering]
            class_ordering = list(class_ordering.reshape(n_classes_orig))
        else:
            class_ordering = list(
                rng.choice(range(n_classes_orig), size=n_classes_orig, replace=False)
            )
        self.rare_classes = class_ordering[:n_rare_classes]
        self.common_classes = class_ordering[
            n_rare_classes : (n_rare_classes + n_common_classes)
        ]

        # The "holdout" classes are always taken from the end of the split, so they
        # are consistent even if n_rare_classes or n_common_classes change.
        holdout_start = len(class_ordering) - n_holdout_classes
        self.holdout_classes = class_ordering[holdout_start:]

        # Define a Zipfian distribution over rare + common classes.
        self.non_holdout_classes = self.rare_classes + self.common_classes
        n_non_holdout = len(self.non_holdout_classes)
        zipf_weights = np.array(
            [1 / j**zipf_exponent for j in range(n_non_holdout, 0, -1)]
        )
        zipf_weights /= np.sum(zipf_weights)
        self.zipf_weights = zipf_weights

        # Save attributes
        self.n_rare_classes = n_rare_classes
        self.n_common_classes = n_common_classes
        self.n_holdout_classes = n_holdout_classes
        self.n_classes = n_rare_classes + n_common_classes + n_holdout_classes
        self.zipf_exponent = zipf_exponent
        self.use_zipf_for_common_rare = use_zipf_for_common_rare
        self.noise_scale = noise_scale

        logging.info(
            "%d rare classes: %s ...", self.n_rare_classes, self.rare_classes[:20]
        )
        logging.info(
            "%d common classes: %s ...", self.n_common_classes, self.common_classes[:20]
        )
        logging.info(
            "%d holdout classes: %s ...",
            self.n_holdout_classes,
            self.holdout_classes[:20],
        )
        logging.info("Zipf exponent: %d", self.zipf_exponent)
        logging.info("Use Zipf for common/rare: %s", self.use_zipf_for_common_rare)
        logging.info("Noise scale: %d", self.noise_scale)

    def _create_noisy_image_seq(self, classes, randomly_generate_rare=False):
        """Return a sequence of images for specified classes, with Gaussian noise added.

        Args:
          classes: a list of the classes, one for each image in the sequence
          randomly_generate_rare: if True, we randomly generate images for the rare
            classes (the same image for all instances of a class, within a
            sequence), rather than using the Omniglot images.

        Returns:
          A numpy array of images, shape (seq_len,H,W,C)
        """
        # TODO(scychan) properly handle non-image data
        classes = np.array(classes)
        if randomly_generate_rare:
            seq_rare_classes = set(classes).intersection(self.rare_classes)
            rare_image_dict = {
                c: np.random.randint(2, size=(IMAGE_SIZE, IMAGE_SIZE, 1))
                for c in seq_rare_classes
            }
            images = np.array(
                [
                    rare_image_dict[c] if c in seq_rare_classes else self.data[c]
                    for c in classes
                ],
                dtype="float32",
            )
        else:
            if isinstance(self.data[classes[0]], list):
                # Randomly sample from the exemplars for each class, without replacement
                images = np.zeros((len(classes), IMAGE_SIZE, IMAGE_SIZE, 1))
                unique_classes = np.unique(classes)
                for c in unique_classes:
                    c_samples = np.random.choice(
                        len(self.data[c]), size=np.sum(classes == c), replace=False
                    )
                    images[classes == c] = np.array(self.data[c])[c_samples]
            else:
                # Just select the single exemplar associated with each class.
                images = np.array([self.data[c] for c in classes])

        # Add pixel noise to the images.
        if self.noise_scale:
            noise = np.random.normal(0, self.noise_scale, images.shape)
            images += noise.astype(images.dtype)
        return images

    def get_bursty_seq(
        self,
        seq_len,
        bursty_shots,
        ways,
        p_bursty,
        p_bursty_common=0.0,
        p_bursty_zipfian=0.0,
        non_bursty_type="common_uniform",
        labeling_common="ordered",
        labeling_rare="unfixed",
        randomly_generate_rare=False,
        grouped=False,
        fs_shots=None,
        p_fewshot=None,
    ):
        """Generate a bursty (or non-bursty) sequence.

        With probability p_bursty, the sequence will contain embedded k-shot n-way
        few-shot problems.
        * Some fraction of these (p_bursty_zipfian) will consist of few-shot
          sequences where the examples are drawn from a Zipfian distribution,
          instead of from distinct common/rare classes.
        * Another fraction of these (p_bursty_common) will consist of few-shot
          sequences of common tokens embedded among rare tokens, with the query
          being one of those common classes.
        * The remaining fraction of these (1 - p_bursty_zipfian - p_bursty_common)
          = p_bursty_rare will consist of few-shot sequences of rare tokens
          embedded among common tokens, with the query being one of those rare
          classes.
        E.g. for shots=2, ways=3, seq_len=9, we might have:
            a C1 b b C2 a a b (a)

        With probability (1-p_bursty), the sequence will contain non-bursty
        sequences -- either Zipfian distributed or uniformly selected from the
        common classes only.

        Args:
          seq_len: number of examples in the sequence, including the query. This
            should be >= shots*ways + 1.
          shots: number of shots, for the few-shot sequences.
          ways: number of ways, for the few-shot sequences.
          p_bursty: probability of a sequence containing a few-shot problem.
          p_bursty_common: fraction of the bursty sequences that are few-shot common
            problems embedded among rare classes (vs. few-shot rare pr oblems
            embedded among common classes)
          p_bursty_zipfian: fraction of bursty sequences that are generated from a
            Zipfian distribution, rather than based on distinct "common" and "rare"
            classes. A common use case is to have p_bursty=1, p_bursty_common=0, and
            p_bursty_zipfian=1 -- in this case there is no distinction between
            common and rare, and all sequences are just few-shot sequences with
            examples drawn from Zipfian distributions. (`labeling_rare` will be used
            for these sequences)
          non_bursty_type: options for the non-bursty sequences: 'zipfian': Drawn
            from the full Zipfian distribution. 'common_uniform': Drawn uniformly
            from common classes. 'common_no_support': No-support seqs from common
            classes.
          labeling_common: how to select the example labels for the common classes
            'ordered': [n_rare_classes:n_classes] (default) or
            [n_rare_classes*X:n_rare_classes*X + n_common_classes] if labeling_rare
            == 'ordered_polysemyX' 'original': use the original Omniglot class
            labels
          labeling_rare: how to select the labels for the rare classes
            'ordered_polysemyX': each example is randomly assigned to one of X
            labels, with X an integer. The labels don't overlap across examples.
            [0:X] for the 1st example, [X:2X] for the 2nd example, etc. 'unfixed':
            randomly assign to [0:n_rare_classes] 'ordered': [0:n_rare_classes]
            'original': use the original Omniglot class labels
          randomly_generate_rare: if True, we randomly generate images for the rare
            classes (the same image for all instances of a class, within a
            sequence), rather than using the Omniglot images.
          grouped: Whether the fewshot sequences (embedded among the remainder) are
            grouped (see get_fewshot_seqs). Note that the remainder can still be
            distribute anywhere, including within the groups.

        Yields:
          A single bursty (or non-bursty) sequence of examples and labels.
        """
        # Process the inputs
        labeling_common = _bytes2str(labeling_common)
        labeling_rare = _bytes2str(labeling_rare)
        non_bursty_type = _bytes2str(non_bursty_type)
        p_bursty_rare = 1 - p_bursty_zipfian - p_bursty_common
        if seq_len < bursty_shots * ways + 1:
            raise ValueError("seq_len must be >= shots * ways + 1")
        generate_remainders = seq_len > bursty_shots * ways + 1
        if "ordered_polysemy" in labeling_rare:
            polysemy_factor = int(labeling_rare.split("ordered_polysemy")[1])
            common_start_idx = self.n_rare_classes * polysemy_factor
            labeling_common = f"ordered{common_start_idx}"
            labeling_rare = f"ordered0_polysemy{polysemy_factor}"

        # Initialize bursty and non-bursty generators.
        if p_bursty < 1:
            if non_bursty_type == "zipfian":
                # Non-bursty sequences are Zipfian distributed.
                supervised_generator = self.get_random_seq(
                    class_type="zipfian",
                    seq_len=seq_len,
                    labeling=labeling_common,
                    randomly_generate_rare=randomly_generate_rare,
                )
            elif non_bursty_type == "common_uniform":
                # Non-bursty sequences are uniformly selected from common classes.
                supervised_generator = self.get_random_seq(
                    class_type="common",
                    seq_len=seq_len,
                    labeling=labeling_common,
                    randomly_generate_rare=randomly_generate_rare,
                )
            elif non_bursty_type == "common_no_support":
                # Non-bursty sequences are no-support sequences of common classes.
                supervised_generator = self.get_no_support_seq(
                    class_type="common",
                    seq_len=seq_len,
                    all_unique=False,
                    labeling=labeling_common,
                    randomly_generate_rare=randomly_generate_rare,
                )
            else:
                raise ValueError(f"Invalid non_bursty_type {non_bursty_type}")
        if p_bursty_rare:
            bursty_rare_generator = self.get_fewshot_seq(
                class_type="rare",
                shots=bursty_shots,
                ways=ways,
                labeling=labeling_rare,
                randomly_generate_rare=randomly_generate_rare,
                grouped=grouped,
            )
            if generate_remainders:
                common_remainder_generator = self.get_random_seq(
                    class_type="common",
                    seq_len=seq_len - bursty_shots * ways - 1,
                    labeling=labeling_common,
                    randomly_generate_rare=randomly_generate_rare,
                )
        if p_bursty_common:
            bursty_common_generator = self.get_fewshot_seq(
                class_type="common",
                shots=bursty_shots,
                ways=ways,
                labeling=labeling_common,
                randomly_generate_rare=randomly_generate_rare,
                grouped=grouped,
            )
            if generate_remainders:
                rare_remainder_generator = self.get_random_seq(
                    class_type="rare",
                    seq_len=seq_len - bursty_shots * ways - 1,
                    labeling=labeling_rare,
                    randomly_generate_rare=randomly_generate_rare,
                )
        if p_bursty_zipfian:
            bursty_zipfian_generator = self.get_fewshot_seq(
                class_type="zipfian",
                shots=bursty_shots,
                ways=ways,
                labeling=labeling_rare,
                randomly_generate_rare=randomly_generate_rare,
                grouped=grouped,
            )
            if generate_remainders:
                zipfian_remainder_generator = self.get_random_seq(
                    class_type="zipfian",
                    seq_len=seq_len - bursty_shots * ways - 1,
                    labeling=labeling_rare,
                    randomly_generate_rare=randomly_generate_rare,
                )

        while True:
            # Determine whether this will be a bursty or non-bursty.
            generate_bursty = random.uniform(0, 1) < p_bursty

            # Generate common-only sequence, if required.
            if not generate_bursty:
                record = next(supervised_generator)

            # Generate bursty sequence, if required.
            else:
                # Determine what type of bursty sequence this will be.
                bursty_determiner = random.uniform(0, 1)
                if bursty_determiner < p_bursty_zipfian:
                    # zipfian
                    bursty_record = next(bursty_zipfian_generator)
                    if generate_remainders:
                        remainder_record = next(zipfian_remainder_generator)
                elif bursty_determiner < p_bursty_common + p_bursty_zipfian:
                    # common
                    bursty_record = next(bursty_common_generator)
                    if generate_remainders:
                        remainder_record = next(rare_remainder_generator)
                else:
                    # rare
                    bursty_record = next(bursty_rare_generator)
                    if generate_remainders:
                        remainder_record = next(common_remainder_generator)

                # Combine them together.
                if generate_remainders:
                    seq_examples = np.concatenate(
                        (remainder_record["example"], bursty_record["example"])
                    )
                    seq_labels = np.concatenate(
                        (remainder_record["label"], bursty_record["label"])
                    )
                    is_rare = np.concatenate(
                        (remainder_record["is_rare"], bursty_record["is_rare"])
                    )
                else:
                    seq_examples = bursty_record["example"]
                    seq_labels = bursty_record["label"]
                    is_rare = bursty_record["is_rare"]

                # Shuffle ordering for all but the last.
                ordering = np.arange(seq_len - 1)
                np.random.shuffle(ordering)
                is_rare[:-1] = is_rare[ordering]
                seq_labels[:-1] = seq_labels[ordering]
                seq_examples[:-1] = seq_examples[ordering]

                record = {
                    "example": seq_examples.astype(np.float32),
                    "label": seq_labels.astype(np.int64),
                    "is_rare": is_rare.astype(np.int64),
                }

            # Alternating labels with zeros, e.g. [label, 0, label, 0, ...].
            zeros = np.zeros_like(record["label"])
            label = record["label"]
            targets = einops.rearrange([label, zeros], "t s -> (s t)")[:-1]
            record["target"] = targets.astype(np.int64)

            # set all elements of target to -1 except the last one
            masked_targets = np.full_like(targets, -1)
            masked_targets[-1] = targets[-1]
            record["targets"] = masked_targets.astype(np.int64)

            yield record

    def get_no_support_seq(
        self,
        class_type,
        seq_len,
        all_unique=True,
        labeling="ordered",
        randomly_generate_rare=False,
    ):
        """Generate a sequence whose support contains no examples of the query class.

        Args:
          class_type: The classes we can sample from ('rare', 'common', 'holdout').
          seq_len: Sequence length.
          all_unique: if True, we generate sequences of all-unique classes.
            Otherwise, the query is first sampled from the distribution
            corresponding to the class_type, and then the support is sampled from
            the remainder of the distribution (with replacement).
          labeling: how to select the labels
            'ordered': [0:n_rare_classes] for the rare examples, and
                       [n_rare_classes:n_classes] for the common examples
            'original': use the original Omniglot class labels
          randomly_generate_rare: if True, we randomly generate images for the rare
            classes (the same image for all instances of a class, within a
            sequence), rather than using the Omniglot images.

        Yields:
          A single sequence of examples and labels.
        """
        class_type = _bytes2str(class_type)
        labeling = _bytes2str(labeling)

        # All-unique generator:
        if all_unique:
            all_unique_generator = self.get_random_seq(
                class_type=class_type,
                seq_len=seq_len,
                labeling=labeling,
                randomly_generate_rare=randomly_generate_rare,
                all_unique=True,
            )
            while True:
                record = next(all_unique_generator)
                yield record

        # Generator that first samples query, then support:
        while True:
            seq_labels = np.zeros(shape=(seq_len), dtype=np.int32)
            if self.example_type == "omniglot":
                seq_examples = np.zeros(
                    shape=(seq_len, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32
                )
            elif self.example_type == "symbolic":
                seq_examples = np.zeros(shape=(seq_len,), dtype=np.float32)

            # Determine which classes we can sample from, and create is_rare sequence.
            classes_to_sample, class_weights = self._get_classes_to_sample(class_type)
            is_rare = self._get_is_rare_seq(class_type, seq_len)

            # Select the query class.
            query_class_idx = np.random.choice(
                range(len(classes_to_sample)), size=1, p=class_weights
            )

            # Select the support classes.
            remaining_class_idx = np.delete(
                range(len(classes_to_sample)), query_class_idx
            )
            remaining_class_weights = np.delete(class_weights, query_class_idx)
            remaining_class_weights /= np.sum(remaining_class_weights)
            support_class_idx = np.random.choice(
                remaining_class_idx,
                size=seq_len - 1,
                replace=True,
                p=remaining_class_weights,
            )
            np.random.shuffle(support_class_idx)

            # Populate the sequence images (with noise).
            seq_class_idx = np.concatenate([support_class_idx, query_class_idx])
            seq_classes = [classes_to_sample[i] for i in seq_class_idx]
            seq_examples[:] = self._create_noisy_image_seq(
                seq_classes, randomly_generate_rare=randomly_generate_rare
            )

            # Populate the sequence labels.
            if labeling == "original":
                seq_labels[:] = seq_classes
            elif labeling == "ordered":
                seq_labels[:] = seq_class_idx
                if class_type == "common":
                    seq_labels += self.n_rare_classes
                elif class_type == "holdout":
                    seq_labels += self.n_rare_classes + self.n_common_classes
            elif "ordered" in labeling:  # 'orderedK'
                seq_labels[:] = seq_class_idx
                label_start = int(labeling.split("ordered")[1])
                seq_labels += label_start
            else:
                return ValueError(f"Invalid value for labeling: {labeling}")

            record = {
                "example": seq_examples,
                "label": seq_labels,
                "is_rare": is_rare,
            }
            yield record

    def get_random_seq(
        self,
        class_type,
        seq_len,
        labeling="ordered",
        randomly_generate_rare=False,
        all_unique=False,
    ):
        """Generate a random sequence of examples.

        Args:
          class_type: The classes we can sample from ('rare', 'common', 'holdout',
            or 'zipfian').
          seq_len: Sequence length.
          labeling: how to select the labels
            'original': use the original Omniglot class labels
            'ordered': [0:n_rare_classes] for the rare examples, and
                       [n_rare_classes:n_classes] for the common examples
            'orderedK': labeled in order [X:n_classes], starting from integer K
          randomly_generate_rare: if True, we randomly generate images for the rare
            classes (the same image for all instances of a class, within a
            sequence), rather than using the Omniglot images.
          all_unique: whether all the examples in a sequence must be unique.

        Yields:
          A single sequence of examples and labels.
        """
        class_type = _bytes2str(class_type)
        labeling = _bytes2str(labeling)

        while True:
            seq_labels = np.zeros(shape=(seq_len), dtype=np.int32)
            if self.example_type == "omniglot":
                seq_examples = np.zeros(
                    shape=(seq_len, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32
                )
            elif self.example_type == "symbolic":
                seq_examples = np.zeros(shape=(seq_len,), dtype=np.float32)

            # Determine which classes we can sample from, and create is_rare sequence.
            classes_to_sample, class_weights = self._get_classes_to_sample(class_type)
            is_rare = self._get_is_rare_seq(class_type, seq_len)

            # Select the query and support classes.
            # (positions 0:seq_len-1 are the support; the last position is the query)
            seq_class_idx = np.random.choice(
                range(len(classes_to_sample)),
                size=seq_len,
                replace=(not all_unique),
                p=class_weights,
            )
            np.random.shuffle(seq_class_idx)

            # Populate the sequence images (with noise).
            seq_classes = [classes_to_sample[i] for i in seq_class_idx]
            seq_examples[:] = self._create_noisy_image_seq(
                seq_classes, randomly_generate_rare=randomly_generate_rare
            )

            # Populate the sequence labels.
            if labeling == "original":
                seq_labels[:] = seq_classes
            elif labeling == "ordered":
                seq_labels[:] = seq_class_idx
                if class_type == "common":
                    seq_labels += self.n_rare_classes
                elif class_type == "holdout":
                    seq_labels += self.n_rare_classes + self.n_common_classes
            elif "ordered" in labeling and "polysemy" not in labeling:  # 'orderedK'
                seq_labels[:] = seq_class_idx
                label_start = int(labeling.split("ordered")[1])
                seq_labels += label_start
            elif "polysemy" in labeling:  # 'orderedK_polysemyX'
                label_start = int(labeling.split("ordered")[1].split("_")[0])
                polysemy_factor = int(labeling.split("polysemy")[1])
                seq_labels[:] = seq_class_idx * polysemy_factor + label_start
                seq_labels[:] += random.choices(range(0, polysemy_factor), k=seq_len)
            else:
                return ValueError(f"Invalid value for labeling: {labeling}")

            record = {
                "example": seq_examples,
                "label": seq_labels,
                "is_rare": is_rare,
            }
            yield record

    def get_fewshot_seq(
        self,
        class_type,
        shots,
        ways,
        labeling="unfixed",
        randomly_generate_rare=False,
        grouped=False,
    ):
        """Generate a sequence whose support is a few-shot training sequence for the query class.

        Args:
          class_type: The classes we can sample from ('rare', 'common', 'holdout',
            or 'zipfian').
          shots: Number of shots (number of examples per class, in the support).
          ways: Number of ways (number of possible classes, per sequence).
          labeling: How labels are selected.
            'orderedK_polysemyX': each example is randomly assigned to one of X
                labels, with X an integer. The labels don't overlap across examples.
                The labels start with integer K.
                I.e. [K:K+X] for 1st example, [K+X:K+2X] for 2nd, etc.
            'unfixed': classes are randomly assigned to 0:ways
            'ordered': [0:n_rare_classes] for the rare examples, and
                       [n_rare_classes:n_classes] for the common examples
            'original': use the original Omniglot class labels
          randomly_generate_rare: if True, we randomly generate images for the rare
            classes (the same image for all instances of a class, within a
            sequence), rather than using the Omniglot images.
          grouped: If True, the examples in the support are grouped, such that every
            k examples contains all k classes. E.g. for 2-shot 2-ways (k=2), we
            could have sequences ABAB, BABA, ABBA, or BAAB.

        Yields:
          A single sequence of examples and labels.
        """
        class_type = _bytes2str(class_type)
        labeling = _bytes2str(labeling)
        seq_len = shots * ways + 1

        while True:
            seq_labels = np.zeros(shape=(seq_len), dtype=np.int32)
            if self.example_type == "omniglot":
                seq_examples = np.zeros(
                    shape=(seq_len, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32
                )
            elif self.example_type == "symbolic":
                seq_examples = np.zeros(shape=(seq_len,), dtype=np.float32)

            # Determine which classes we can sample from, and create is_rare sequence.
            classes_to_sample, class_weights = self._get_classes_to_sample(class_type)
            is_rare = self._get_is_rare_seq(class_type, seq_len)

            # Select n classes for the sequence.
            # "class" refers to the key for an example in self.data.
            # "label" refers to the label that a model will be expected to predict.
            if "polysemy" in labeling:  # orderedK_polysemyX
                label_start = int(labeling.split("ordered")[1].split("_")[0])
                polysemy_factor = int(labeling.split("polysemy")[1])
                class_options_idx = np.random.choice(
                    range(len(classes_to_sample)),
                    size=ways,
                    replace=True,
                    p=class_weights,
                )
                class_options = [classes_to_sample[i] for i in class_options_idx]
                label_options = np.array(class_options_idx) * polysemy_factor
                label_options += random.choices(range(0, polysemy_factor), k=ways)
                label_options += label_start
                label_options = list(label_options)
            elif labeling == "unfixed":
                label_options = list(range(ways))
                class_options = list(
                    np.random.choice(
                        classes_to_sample, size=ways, replace=True, p=class_weights
                    )
                )
            elif labeling == "ordered":
                class_options_idx = np.random.choice(
                    range(len(classes_to_sample)),
                    size=ways,
                    replace=True,
                    p=class_weights,
                )
                class_options = [classes_to_sample[i] for i in class_options_idx]
                label_options = class_options_idx.tolist()
                if class_type == "common":
                    label_options = [l + self.n_rare_classes for l in label_options]
                elif class_type == "holdout":
                    label_options = [
                        l + self.n_classes - self.n_holdout_classes
                        for l in label_options
                    ]
            elif labeling == "original":
                label_options = list(
                    np.random.choice(
                        classes_to_sample, size=ways, replace=True, p=class_weights
                    )
                )
                class_options = label_options
            else:
                raise ValueError("Invalid value for labeling: %s" % labeling)

            # Select one class for the query.
            query_idx = random.choice(range(ways))
            query_label = label_options[query_idx]
            query_class = class_options[query_idx]

            # Get the labels and examples for the few-shot sequence.
            seq_labels[:] = label_options * shots + [query_label]
            seq_classes = class_options * shots + [query_class]
            seq_examples = self._create_noisy_image_seq(
                seq_classes, randomly_generate_rare=randomly_generate_rare
            )

            # Shuffle ordering.
            ordering = np.arange(seq_len - 1)
            if grouped:
                for i in range(shots):
                    np.random.shuffle(ordering[i * ways : (i + 1) * ways])
            else:
                np.random.shuffle(ordering)
            is_rare[:-1] = is_rare[ordering]
            seq_labels[:-1] = seq_labels[ordering]
            seq_examples[:-1] = seq_examples[ordering]

            record = {
                "example": seq_examples.astype(np.float32),
                "label": seq_labels.astype(np.int64),
                "is_rare": is_rare.astype(np.int64),
            }

            # Alternating labels with zeros, e.g. [label, 0, label, 0, ...].
            zeros = np.zeros_like(record["label"])
            label = record["label"]
            targets = einops.rearrange([label, zeros], "t s -> (s t)")[:-1]
            record["target"] = targets

            # set all elements of target to -1 except the last one
            masked_targets = np.full_like(targets, -1)
            masked_targets[-1] = targets[-1]
            record["targets"] = masked_targets

            yield record

    def get_mixed_seq(self, shots, ways, p_fewshot):
        """Generate either a few-shot or supervised sequence.

        * Few-shot sequences consist of rare classes only, with labels randomly
        assigned [0:ways].
        * Supervised sequences consist of common classes only, with labels fixed
        in the range [n_rare_classes:total_n_classes].
        NB: the labels [ways:n_rare_classes] may be unused.

        Args:
          shots: Number of shots (number of examples per class, in the support).
          ways: Number of ways (number of possible classes, per sequence).
          p_fewshot: Probability of a sequence being few-shot rare (vs supervised
            common).

        Yields:
          A single sequence of examples and labels.
        """

        # Initialize generators for no-support-common and few-shot-rare.
        supervised_generator = self.get_random_seq(
            class_type="common",
            seq_len=(shots * ways + 1),
            labeling="ordered",
            randomly_generate_rare=False,
            all_unique=False,
        )
        fewshot_generator = self.get_fewshot_seq(
            class_type="rare", shots=shots, ways=ways, randomly_generate_rare=False
        )

        # Randomly yield from each generator, according to the proportion
        while True:
            generate_fewshot = random.uniform(0, 1) < p_fewshot
            if generate_fewshot:
                record = next(fewshot_generator)
            else:
                record = next(supervised_generator)
            yield record

    def _get_classes_to_sample(self, class_type):
        """Given a class type, returns a list of classes and their weights."""
        if class_type == "rare":
            classes_to_sample = self.rare_classes
        elif class_type == "common":
            classes_to_sample = self.common_classes
        elif class_type == "holdout":
            classes_to_sample = self.holdout_classes
        elif class_type == "zipfian":
            classes_to_sample = self.non_holdout_classes
        else:
            raise ValueError(f"Invalid value for class_type: {class_type}")

        if class_type == "zipfian":
            class_weights = self.zipf_weights
        elif self.use_zipf_for_common_rare and class_type in ["common", "rare"]:
            if class_type == "common":
                class_weights = self.zipf_weights[self.n_rare_classes :]
            elif class_type == "rare":
                class_weights = self.zipf_weights[: self.n_rare_classes]
            class_weights /= np.sum(class_weights)
        else:
            n_classes = len(classes_to_sample)
            class_weights = np.full(n_classes, 1 / n_classes)
        return classes_to_sample, class_weights

    def _get_is_rare_seq(self, class_type, seq_len):
        if class_type == "rare":
            is_rare = np.ones(seq_len, dtype=np.int32)
        elif class_type in ("common", "holdout", "zipfian"):
            is_rare = np.zeros(seq_len, dtype=np.int32)
        else:
            raise ValueError(f"Invalid value for class_type: {class_type}")
        return is_rare


def _bytes2str(x):
    """Convert bytes to str, if needed."""
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    return x


# stuff for embedder


def _create_positional_encodings(inputs, max_time=30.0):
    """Generates positional encodings for the input.

    Args:
      inputs: A tensor of shape [batch_size, seq_len, emb_size].
      max_time: (default 10000) Constant used to scale position by in the
        encodings.

    Returns:
      pos_emb: as defined above, of size [1, seq_len, emb_size].
    """

    _, seq_len, embedding_size = inputs.shape

    if embedding_size % 2 == 1:
        raise ValueError("Embedding sizes must be even if using positional encodings.")

    # Generate a sequence of positions and frequencies.
    # pos = np.arange(seq_len, dtype=np.float32)
    pos = torch.arange(seq_len, device=inputs.device)
    # freqs = np.arange(0, embedding_size, 2, dtype=np.float32)
    freqs = torch.arange(0, embedding_size, 2, device=inputs.device)
    inverse_freqs = 1.0 / (max_time ** (freqs / embedding_size))

    # We combine [seq_len] and [emb_size / 2] to [seq_len, emb_size / 2].
    # pos_emb = np.einsum("i,j->ij", pos, inverse_freqs)
    pos_emb = torch.einsum("i,j->ij", pos, inverse_freqs)

    # Concat sines and cosines and return.
    # pos_emb = np.concatenate([np.sin(pos_emb), np.cos(pos_emb)], -1)
    pos_emb = torch.concatenate([torch.sin(pos_emb), torch.cos(pos_emb)], -1)

    return pos_emb


class OmniglotDataset(IterableDataset):
    def __init__(self, seq_generator, generator_args, generators, seed):
        self.seq_generator = seq_generator
        self.generator_args = generator_args
        self.seed = seed
        self.generators = generators

    def reset(self):
        pass

    def __iter__(self):
        return self.seq_generator(**self.generator_args)


class OmniglotEmbedder(nn.Module):
    """Input embedder."""

    def __init__(
        self,
        cfg,
        name=None,  # TODO: do we still need this?
    ):
        """Initialize the input embedder.

        Args:
          num_classes: Total number of output classes.
          emb_dim: Dimensionality of example and label embeddings.
          example_encoding: How to encode example inputs.
            'resnet': simple resnet encoding
            'linear': flatten and pass through a linear layer
            'embedding': pass through an embedding layer
          flatten_superpixels: Whether to flatten the output of the resnet (instead
            of taking a mean over superpixels).
          example_dropout_prob: Dropout probability on example embeddings. Note that
            these are applied at both train and test.
          concatenate_labels: Whether to concatenate example and label embeddings
            into one token for each (example, label) pair, rather than being fed to
            the transformer as two separate tokens.
          use_positional_encodings: Whether to use positional encoding.
          positional_dropout_prob: Positional dropout probability.
          name: Optional name for the module.
        """
        # super(OmniglotInputEmbedder, self).__init__(name=name)
        super().__init__()
        self._num_classes = cfg.data.num_classes
        self._emb_dim = cfg.model.d_model
        self._example_encoding = cfg.embedder.example_encoding
        self._flatten_superpixels = cfg.embedder.flatten_superpixels
        self._example_dropout_prob = cfg.embedder.example_dropout_prob
        self._concatenate_labels = cfg.embedder.concatenate_labels
        self._use_positional_encodings = cfg.embedder.use_positional_encodings
        self._positional_dropout_prob = cfg.embedder.positional_dropout_prob
        self.merge_embeds = cfg.train.merge_embeds
        self.merge_type = cfg.train.merge_type

        # initialize resnet to embed images
        resnet_config = transformers.AutoConfig.from_pretrained("microsoft/resnet-18")
        """
        The images were embedded by a ResNet with two blocks per group and
        channels per group (16, 32, 32, 64), and which was not pretrained.
        The integer labels were embedded using a standard embedding layer.
        """
        if self.merge_embeds and self.merge_type == "concat":
            d_resnet = self._emb_dim // 2
        else:
            d_resnet = self._emb_dim
        resnet_config.update(
            {
                "depths": [2, 2, 2, 2],
                "hidden_sizes": [16, 32, 32, d_resnet],
                "num_channels": 1,
            }
        )
        self.example_encoding = transformers.ResNetModel(resnet_config)

        # initialize embedding layer to embed labels
        if self.merge_embeds and self.merge_type == "concat":
            self.embs = torch.nn.Embedding(self._num_classes, self._emb_dim // 2)
        else:
            self.embs = torch.nn.Embedding(self._num_classes, self._emb_dim)
        torch.nn.init.trunc_normal_(self.embs.weight, std=0.02)

    def forward(self, batch: dict):
        images = batch["example"]
        labels = batch["label"]
        batch_sz = images.shape[0]
        images = einops.rearrange(images, "b s h w c -> (b s) c h w")
        # embed images with resnet
        h_images = self.example_encoding(images, return_dict=True).pooler_output
        h_images = einops.rearrange(h_images, "(b s) e 1 1 -> b s e", b=batch_sz)
        h_label = self.embs(labels)  # [b, s, e]
        # Interleave example and label embeddings (note s before t)
        hh = einops.rearrange([h_images, h_label], "t b s e -> b (s t) e")

        if self.merge_embeds:
            a = hh[:, ::2, :]
            b = hh[:, 1::2, :]
            # mask the last element of b by setting to zero
            b[:, -1, :] = 0
            if self.merge_type == "sum":
                hh = torch.add(a, b)
            elif self.merge_type == "concat":
                hh = torch.cat((a, b), dim=-1)
            else:
                raise ValueError("Invalid merge type")
        else:
            # drop the final label since we are predicting it
            hh = hh[:, :-1, :]

        # Create positional encodings.
        if self._use_positional_encodings:
            foo  # sanity check that we aren't using pos emb here
            positional_encodings = _create_positional_encodings(hh)
            # if is_training:
            #     positional_encodings = hk.dropout(
            #         hk.next_rng_key(),
            #         self._positional_dropout_prob,
            #         positional_encodings,
            #     )
            # Add on the positional encoding.
            hh += positional_encodings

        return hh


class OmniglotHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.model.d_model, cfg.data.num_classes)

    def forward(self, x):
        return self.fc(x)


def _omniglot_loss_fn(logits, targets, batch):
    assert logits.size(1) == 1 or logits.size(1) == targets.size(1)
    if logits.size(1) == 1:
        targets = targets[:, -1:]
    cel = nn.CrossEntropyLoss(ignore_index=IGNORE_IDX, reduction="none")
    batch_size = logits.shape[0]
    logits = rearrange(logits, "b s c -> (b s) c")
    targets = rearrange(targets, "b s -> (b s)")
    loss = cel(logits, targets)
    loss = rearrange(loss, "(b s) -> b s", b=batch_size)
    loss = reduce(loss, "b s -> s", reduction="mean")
    return loss


def omniglot_loss_fn(logits, targets, batch, foo=None):
    logits = logits[:, -1, :]  # [b, vocab_size]
    targets = targets[:, -1]  # [b]
    loss = F.cross_entropy(logits, targets, reduction="none")
    loss = rearrange(loss, "b -> b 1")
    return loss
