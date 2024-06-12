"""Module containing class for storing and looking up association scores."""

import torch
import numpy as np
import pandas as pd
import attrs
from dreem.io import Instance


@attrs.define
class AssociationMatrix:
    """Class representing the associations between detections.

    Attributes:
        matrix: the `n_query x n_ref` association matrix`
        ref_instances: all instances used to associate against.
        query_instances: query instances that were associated against ref instances.
    """

    matrix: np.ndarray | torch.Tensor
    ref_instances: list[Instance] = attrs.field()
    query_instances: list[Instance] = attrs.field()

    @ref_instances.validator
    def _check_ref_instances(self, attribute, value):
        """Check to ensure that the number of association matrix columns and reference instances match.

        Args:
            attribute: The ref instances.
            value: the list of ref instances.

        Raises:
            ValueError if the number of columns and reference instances don't match.
        """
        if len(value) != self.matrix.shape[-1]:
            raise ValueError(
                (
                    "Ref instances must equal number of columns in Association matrix"
                    f"Found {len(value)} ref instances but {self.matrix.shape[-1]} columns."
                )
            )

    @query_instances.validator
    def _check_query_instances(self, attribute, value):
        """Check to ensure that the number of association matrix rows and query instances match.

        Args:
            attribute: The query instances.
            value: the list of query instances.

        Raises:
            ValueError if the number of rows and query instances don't match.
        """
        if len(value) != self.matrix.shape[0]:
            raise ValueError(
                (
                    "Query instances must equal number of rows in Association matrix"
                    f"Found {len(value)} query instances but {self.matrix.shape[0]} rows."
                )
            )

    def __repr__(self) -> str:
        """Get the string representation of the Association Matrix.

        Returns:
            the string representation of the association matrix.
        """
        return (
            f"AssociationMatrix({self.matrix},"
            f"query_instances={len(self.query_instances)},"
            f"ref_instances={len(self.ref_instances)})"
        )

    def numpy(self) -> np.ndarray:
        """Convert association matrix to a numpy array.

        Returns:
            The association matrix as a numpy array.
        """
        if isinstance(self.matrix, torch.Tensor):
            return self.matrix.detach().cpu().numpy()
        return self.matrix

    def to_dataframe(
        self, row_labels: str = "gt", col_labels: str = "gt"
    ) -> pd.DataFrame:
        """Convert the association matrix to a pandas DataFrame.

        Args:
            row_labels: How to label the rows(queries).
                If list, then must match # of rows/queries
                If `"gt"` then label by gt track id.
                If `"pred"` then label by pred track id.
                Otherwise label by the query_instance indices
            col_labels: How to label the columns(references).
                If list, then must match # of columns/refs
                If `"gt"` then label by gt track id.
                If `"pred"` then label by pred track id.
                Otherwise label by the ref_instance indices

        Returns:
            The association matrix as a pandas dataframe.
        """
        matrix = self.numpy()

        if not isinstance(row_labels, str):
            if len(row_labels) == len(self.query_instances):
                row_inds = row_labels

            else:
                raise ValueError(
                    (
                        f"Mismatched # of rows and labels!",
                        f"Found {len(row_labels)} with {len(self.query_instances)} rows",
                    )
                )

        else:
            if row_labels == "gt":
                row_inds = [
                    instance.gt_track_id.item() for instance in self.query_instances
                ]

            elif row_labels == "pred":
                row_inds = [
                    instance.pred_track_id.item() for instance in self.query_instances
                ]

            else:
                row_inds = np.arange(len(self.query_instances))

        if not isinstance(col_labels, str):
            if len(col_labels) == len(self.ref_instances):
                col_inds = col_labels

            else:
                raise ValueError(
                    (
                        f"Mismatched # of columns and labels!",
                        f"Found {len(col_labels)} with {len(self.ref_instances)} columns",
                    )
                )

        else:
            if col_labels == "gt":
                col_inds = [
                    instance.gt_track_id.item() for instance in self.ref_instances
                ]

            elif col_labels == "pred":
                col_inds = [
                    instance.pred_track_id.item() for instance in self.ref_instances
                ]

            else:
                col_inds = np.arange(len(self.ref_instances))

        asso_df = pd.DataFrame(matrix, index=row_inds, columns=col_inds)

        return asso_df

    def reduce(
        self,
        row_dims: str = "instance",
        col_dims: str = "track",
        row_grouping: str | None = None,
        col_grouping: str = "pred",
        reduce_method: callable = np.sum,
    ) -> pd.DataFrame:
        """Aggregate the association matrix by specified dimensions and grouping.

        Args:
           row_dims: A str indicating how to what dimensions to reduce rows to.
                Either "instance" (remains unchanged), or "track" (n_rows=n_traj).
           col_dims: A str indicating how to dimensions to reduce rows to.
                Either "instance" (remains unchanged), or "track" (n_cols=n_traj)
           row_grouping: A str indicating how to group rows when aggregating. Either "pred" or "gt".
           col_grouping: A str indicating how to group columns when aggregating. Either "pred" or "gt".
           reduce_method: A callable function that operates on numpy matrices and can take an `axis` arg for reducing.

        Returns:
            The association matrix reduced to an inst/traj x traj/inst association matrix as a dataframe.
        """
        n_rows = len(self.query_instances)
        n_cols = len(self.ref_instances)

        col_tracks = {-1: self.ref_instances}
        row_tracks = {-1: self.query_instances}

        col_inds = [i for i in range(len(self.ref_instances))]
        row_inds = [i for i in range(len(self.query_instances))]

        if col_dims == "track":
            col_tracks = self.get_tracks(self.ref_instances, col_grouping)
            col_inds = list(col_tracks.keys())
            n_cols = len(col_inds)

        if row_dims == "track":
            row_tracks = self.get_tracks(self.query_instances, row_grouping)
            row_inds = list(row_tracks.keys())
            n_rows = len(row_inds)

        reduced_matrix = []
        for row_track, row_instances in row_tracks.items():
            for col_track, col_instances in col_tracks.items():

                asso_matrix = self[row_instances, col_instances]

                if col_dims == "track":
                    asso_matrix = reduce_method(asso_matrix, axis=1)

                if row_dims == "track":
                    asso_matrix = reduce_method(asso_matrix, axis=0)

                reduced_matrix.append(asso_matrix)

        reduced_matrix = np.array(reduced_matrix).reshape(n_cols, n_rows).T

        return pd.DataFrame(reduced_matrix, index=row_inds, columns=col_inds)

    def __getitem__(
        self, inds: tuple[int | Instance | list[int | Instance]]
    ) -> np.ndarray:
        """Get elements of the association matrix.

        Args:
            inds: A tuple of query indices and reference indices.
                Indices can be either:
                    A single instance or integer.
                    A list of instances or integers.

        Returns:
            An np.ndarray containing the elements requested.
        """
        query_inst, ref_inst = inds

        query_ind = self.__getindices__(query_inst, self.query_instances)
        ref_ind = self.__getindices__(ref_inst, self.ref_instances)

        try:
            return self.numpy()[query_ind[:, None], ref_ind].squeeze()
        except IndexError as e:
            print(f"Query_insts: {type(query_inst)}")
            print(f"Query_inds: {query_ind}")
            print(f"Ref_insts: {type(ref_inst)}")
            print(f"Ref_ind: {ref_ind}")
            raise (e)

    def __getindices__(
        self,
        instance: Instance | int | np.typing.ArrayLike,
        instance_lookup: list[Instance],
    ) -> np.ndarray:
        """Get the indices of the instance for lookup.

        Args:
            instance: The instance(s) to be retrieved
                Can either be a single int/instance or a list of int/instances
            instance_lookup: A list of Instances to be used to retrieve indices

        Returns:
            A np array of indices.
        """
        if isinstance(instance, Instance):
            ind = np.array([instance_lookup.index(instance)])

        elif instance is None:
            ind = np.arange(len(instance_lookup))

        elif np.isscalar(instance):
            ind = np.array([instance])

        else:
            instances = instance
            if not [isinstance(inst, (Instance, int)) for inst in instance]:
                raise ValueError(
                    f"List of indices must be `int` or `Instance`. Found {set([type(inst) for inst in instance])}"
                )
            ind = np.array(
                [
                    (
                        instance_lookup.index(instance)
                        if isinstance(instance, Instance)
                        else instance
                    )
                    for instance in instances
                ]
            )

        return ind

    def get_tracks(
        self, instances: list["Instance"], label: str = "pred"
    ) -> dict[int, list["Instance"]]:
        """Group instances by track.

        Args:
            instances: The list of instances to group
            label: the track id type to group by. Either `pred` or `gt`.

        Returns:
            A dictionary of track_id:instances
        """
        if label == "pred":
            traj_ids = set([instance.pred_track_id.item() for instance in instances])
            traj = {
                track_id: [
                    instance
                    for instance in instances
                    if instance.pred_track_id.item() == track_id
                ]
                for track_id in traj_ids
            }

        elif label == "gt":
            traj_ids = set(
                [instance.gt_track_id.item() for instance in self.ref_instances]
            )
            traj = {
                track_id: [
                    instance
                    for instance in self.ref_instances
                    if instance.gt_track_id.item() == track_id
                ]
                for track_id in traj_ids
            }

        else:
            raise ValueError(f"Unsupported label '{label}'. Expected 'pred' or 'gt'.")

        return traj
