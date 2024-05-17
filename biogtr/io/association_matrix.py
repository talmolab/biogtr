"""Module containing class for storing and looking up association scores."""

import torch
import numpy as np
import pandas as pd
import attrs
from biogtr.io import Instance
from typing import Union


@attrs.define
class AssociationMatrix:
    """Class representing the associations between detections.

    Attributes:
        matrix: the `n_query x n_ref` association matrix`
        ref_instances: all instances used to associate against.
        query_instances: query instances that were associated against ref instances.
    """

    matrix: Union[np.ndarray, torch.Tensor] = attrs.field()
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
                    f"Found {len(value)} query instances but {self.matrix.shape[0]} columns."
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
        self, row_label: str = "gt", col_label: str = "gt"
    ) -> pd.DataFrame:
        """Convert the association matrix to a pandas DataFrame.

        Args:
            row_label: How to label the rows(queries).
                        If `gt` then label by gt track id.
                        If `pred` then label by pred track id.
                        Otherwise label by the query_instance indices
            col_label: How to label the columns(references).
                        If `gt` then label by gt track id.
                        If `pred` then label by pred track id.
                        Otherwise label by the ref_instance indices

        Returns:
            The association matrix as a pandas dataframe.
        """
        matrix = self.numpy()

        if row_label.lower() == "gt":
            row_inds = [
                instance.gt_track_id.item() for instance in self.query_instances
            ]

        elif row_label.lower() == "pred":
            row_inds = [
                instance.pred_track_id.item() for instance in self.query_instances
            ]

        else:
            row_inds = np.arange(len(self.query_instances))

        if col_label.lower() == "gt":
            col_inds = [instance.gt_track_id.item() for instance in self.ref_instances]

        elif col_label.lower() == "pred":
            col_inds = [
                instance.pred_track_id.item() for instance in self.ref_instances
            ]

        else:
            col_inds = np.arange(len(self.ref_instances))

        asso_df = pd.DataFrame(matrix, index=row_inds, columns=col_inds)

        return asso_df

    def __getitem__(self, inds) -> np.ndarray:
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
        instance: Union[Instance, int, np.typing.ArrayLike],
        instance_lookup: list[Instance],
    ) -> np.ndarray:
        """Get the indices of the instance for lookup.

        Args:
            instance: The instance(s) to be retrieved
                      Can either be a single int/instance or a list of int/instances/
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