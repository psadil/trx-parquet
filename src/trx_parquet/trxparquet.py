import dataclasses
import logging
import os
import random
import string
from pathlib import Path
from typing import Collection, Literal, TypeAlias

import nibabel as nb
import numpy as np
import numpy.typing as npt
import polars as pl
import polars.selectors as cs
import pyarrow.dataset as ds
from dipy.io import stateful_tractogram
from nibabel import affines
from pyarrow import parquet as pq
from trx import trx_file_memmap

VOXEL_ORDER: TypeAlias = Literal[
    "RAS", "LAS", "RPS", "LPS", "RAI", "LAI", "RPI", "LPI"
]


def randomword(length: int) -> str:
    """Generate Random Word of Given Length From Lower Case ASCII

    Args:
        length : Length of random word to generate

    Returns:
        str: String of given length (not safe for cryptographic purposes)
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def init_example_trxparquet(
    n_streamlines: int,
    n_points_per_streamline: int,
    n_types_data_per_streamline: int = 0,
    n_types_data_per_point: int = 0,
    dimensions: npt.NDArray[np.uint16] | None = None,
    voxel_to_rasmm: npt.NDArray[np.float32] | None = None,
) -> "TrxParquet":
    """Create trxparquet for experimentation

    Args:
        n_streamlines : Number of streamlines in data.
        n_points_per_streamline : Number of points per streamline.
        n_types_data_per_streamline : Number of scalar datatypes to associate
            with each streamline. Defaults to 0.
        n_types_data_per_point : Number of scalar datatypes to associate
            with each point. Defaults to 0.
        dimensions : Voxel dimensions used during header creation.
            Defaults to None, which implies (20, 20, 20).
        voxel_to_rasmm : Affine used during header creation.
            Defaults to None, which implies scaled identity.

    Returns:
        TrxParquet: Instance of TrxParquet
    """
    if dimensions is None:
        dimensions = np.array((20, 20, 20), dtype=np.uint16)
    if voxel_to_rasmm is None:
        voxel_to_rasmm = np.diag(np.append(dimensions, 1)).astype(
            dtype=np.float32
        )

    header = TrxHeader(DIMENSIONS=dimensions, VOXEL_TO_RASMM=voxel_to_rasmm)
    n_rows = n_streamlines * n_points_per_streamline

    data = pl.DataFrame(
        data=np.random.uniform(size=(n_rows, 3)),
        schema=[f"protected_position_{i}" for i in range(3)],
    ).with_columns(
        pl.Series(
            "protected_streamline",
            values=np.repeat(
                range(n_streamlines), repeats=n_points_per_streamline
            ),
        )
    )

    if n_types_data_per_streamline > 0:
        dps = pl.DataFrame(
            np.random.uniform(
                size=(n_streamlines, n_types_data_per_streamline)
            ),
            schema=[
                f"dps_{randomword(6)}"
                for _ in range(n_types_data_per_streamline)
            ],
        ).with_columns(
            protected_streamline=pl.Series(np.arange(n_streamlines))
        )
        data = data.join(dps, on="protected_streamline")
    if n_types_data_per_point > 0:
        dpv = pl.DataFrame(
            np.random.uniform(size=(n_rows, n_types_data_per_point)),
            schema=[
                f"dpv_{randomword(6)}" for _ in range(n_types_data_per_point)
            ],
        )
        data = pl.concat([data, dpv], how="horizontal")

    return TrxParquet.from_data_header(header=header, data=data)


def np_to_pl_dtype(dtype: np.dtype) -> pl.PolarsDataType:
    if dtype == np.float32:
        out = pl.Float32
    elif dtype == np.uint32:
        out = pl.UInt32
    elif dtype == np.uint16:
        out = pl.UInt16
    else:
        msg = "Unable to find matching dtype"
        raise ValueError(msg)

    return out


@dataclasses.dataclass(eq=True, frozen=True)
class TrxHeader:
    DIMENSIONS: npt.NDArray[np.uint16]
    VOXEL_TO_RASMM: npt.NDArray[np.float32]

    def to_bytes_dict(self) -> dict[bytes, bytes]:
        metadata = {
            b"DIMENSIONS": self.DIMENSIONS.tobytes(),
            b"VOXEL_TO_RASMM": self.VOXEL_TO_RASMM.tobytes(),
        }
        return metadata

    @classmethod
    def from_parquet_metadata(
        cls, metadata: dict[bytes, bytes]
    ) -> "TrxHeader":
        if _dimensions := metadata.get(b"DIMENSIONS"):
            dimensions = cls.parse_dimensions(_dimensions)
        else:
            msg = "Unable to find DIMENSIONS in metadata"
            raise AssertionError(msg)

        if _voxel_to_rasmm := metadata.get(b"VOXEL_TO_RASMM"):
            voxel_to_rasmm = cls.parse_voxel_to_rasmm(_voxel_to_rasmm)
        else:
            msg = "Unable to find VOXEL_TO_RASMM in metadata"
            raise AssertionError(msg)

        return cls(DIMENSIONS=dimensions, VOXEL_TO_RASMM=voxel_to_rasmm)

    @classmethod
    def from_nifti(
        cls, nifti: nb.nifti1.Nifti1Image | nb.nifti1.Nifti1Pair
    ) -> "TrxHeader":
        return cls(
            DIMENSIONS=np.ndarray(nifti.shape), VOXEL_TO_RASMM=nifti.affine
        )

    @staticmethod
    def parse_n_dps(n_dps: bytes) -> npt.NDArray[np.uint32]:
        return np.frombuffer(n_dps, dtype=np.uint32)

    @staticmethod
    def parse_n_dpv(n_dpb: bytes) -> npt.NDArray[np.uint32]:
        return np.frombuffer(n_dpb, dtype=np.uint32)

    @staticmethod
    def parse_dimensions(dimensions: bytes) -> npt.NDArray[np.uint16]:
        return np.frombuffer(dimensions, dtype=np.uint16)

    @staticmethod
    def parse_voxel_to_rasmm(voxel_to_rasmm: bytes) -> npt.NDArray[np.float32]:
        return np.frombuffer(voxel_to_rasmm, dtype=np.float32).reshape(4, 4)


@dataclasses.dataclass
class TrxParquet:
    """Core class of the TrxParquet"""

    header: TrxHeader
    data: pl.DataFrame | pl.LazyFrame

    @property
    def streamlines(self) -> stateful_tractogram.Streamlines:
        if isinstance(self.data, pl.LazyFrame):
            data = self.data.collect()
        else:
            data = self.data
        streamlines = (
            data.select(
                cs.matches("protected_streamline"),
                cs.starts_with("protected_position"),
            )
            .group_by("protected_streamline")
            .all()
            .select(cs.starts_with("protected_position"))
            .to_numpy()
        )
        return stateful_tractogram.Streamlines(
            np.vstack(s).T for s in streamlines
        )

    @property
    def n_streamlines(self) -> int:
        unique = self.data.select("protected_streamline").unique()
        if isinstance(unique, pl.LazyFrame):
            unique = unique.collect()
        return unique.shape[0]

    @property
    def data_per_vertex(
        self,
    ) -> dict[str, nb.streamlines.array_sequence.ArraySequence] | None:
        """Data Per Vertex.

        Notes:
            This function will cause duplication of data in dpv_ columns of
                data attribute

        Returns:
            dict[str, nb.streamlines.array_sequence.ArraySequence] | None:
                Data Per Vertex.
        """
        data = (
            self.data.select("protected_streamline", cs.starts_with("dpv"))
            .group_by("protected_streamline", maintain_order=True)
            .all()
            .select(cs.starts_with("dpv"))
        )
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        if data.shape[0] == 0:
            dpv = None
        else:
            dpv = {
                k.removeprefix(
                    "dpv_"
                ): nb.streamlines.array_sequence.ArraySequence(
                    np.array(v.to_list())  # NOTE: to_list copies the data
                )
                for k, v in data.to_dict().items()
            }
        return dpv

    @property
    def data_per_streamline(self) -> dict[str, npt.NDArray] | None:
        """Data Per Streamline.

        Notes:
            This function should not cause data duplication.

        Returns:
            dict[str, npt.NDArray] | None: Data Per Streamline
        """
        data = (
            self.data.group_by("protected_streamline")
            .head(1)
            .select(cs.starts_with("dps"))
        )
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        if data.shape[0] == 0:
            dps = None
        else:
            dps = {
                k.removeprefix("dps_"): v.to_numpy(zero_copy_only=True)
                for k, v in data.to_dict().items()
            }
        return dps

    @classmethod
    def from_data_header(
        cls, header: TrxHeader, data: pl.DataFrame
    ) -> "TrxParquet":
        """Initalize TrxParquet from header and data pair

        Args:
            header: Header of object to initialize
            data: Data attribute

        Notes:
            This function exists mostly to ensure column ordering in data

        Returns:
            TrxParquet: instance
        """
        data = data.select(
            "protected_streamline",
            cs.starts_with("protected_position"),
            cs.starts_with("dps"),
            cs.starts_with("dpv"),
        )
        return cls(header=header, data=data)

    @classmethod
    def from_trx_file(cls, src: os.PathLike) -> "TrxParquet":
        """Build Parquet-style TrxParquet from .trx

        Args:
            src: Location of file to read.

        Returns:
            TrxParquet: Instance of TrxParquet
        """

        # TODO: add data_per_group

        trxfile = trx_file_memmap.load(str(src))
        if trxfile.groups:
            logging.warning(
                "Detected groups in src, but groups not implemented. They will be ignored."
            )

        positions = pl.from_numpy(
            trxfile.streamlines._data,  # type: ignore
            schema=[
                "protected_position_0",
                "protected_position_1",
                "protected_position_2",
            ],
        )

        _dps = []
        for k, v in trxfile.data_per_streamline.items():
            _dps.append(
                pl.DataFrame(
                    {
                        f"dps_{k}": pl.Series(
                            values=v,
                            dtype=pl.Array(
                                width=v.shape[1], inner=np_to_pl_dtype(v.dtype)
                            ),
                        )
                    }
                )
            )

        dps = pl.concat(_dps, how="horizontal").with_columns(
            pl.Series(
                "protected_streamline",
                range(0, trxfile.header.get("NB_STREAMLINES")),  # type: ignore
            )
        )

        _dpv: list[pl.DataFrame] = []
        _streamlines = []
        for i, (k, v) in enumerate(trxfile.data_per_vertex.items()):
            _dpv_dfs = []
            for s, streamline in enumerate(v):
                _dpv_dfs.append(
                    pl.DataFrame(
                        pl.Series(
                            name=f"dpv_{k}",
                            dtype=pl.Array(
                                width=streamline.shape[1],
                                inner=np_to_pl_dtype(v[0].dtype),
                            ),
                            values=streamline,
                        )
                    )
                )
                if i == 0:
                    _streamlines.extend([s] * streamline.shape[0])

            _dpv.append(pl.concat(_dpv_dfs))

        dpv = (
            pl.concat(_dpv, how="horizontal")
            .with_columns(protected_streamline=pl.Series(_streamlines))
            .join(dps, on="protected_streamline")
            .hstack(positions)
        )

        metadata = {}
        metadata["DIMENSIONS"] = trxfile.header.get("DIMENSIONS")
        metadata["VOXEL_TO_RASMM"] = trxfile.header.get("VOXEL_TO_RASMM")
        header = TrxHeader(**metadata)

        return TrxParquet.from_data_header(header=header, data=dpv)

    @classmethod
    def from_file(
        cls,
        src: os.PathLike,
        loadtype: Literal["lazy", "memory", "memory_map"] = "lazy",
        streamlines: Collection[pl.UInt64] | None = None,
    ) -> "TrxParquet":
        """Read TrxParquet from Parquet file.

        Args:
            src : Location of parquet file to read
            loadtype (Literal[&quot;lazy&quot;, &quot;memory&quot;, &quot;memory_map&quot;], optional):
                Manner in which data should be loaded by polars. Defaults to "lazy".
            streamlines : Filter used for selection of subset of streamlines.

        Raises:
            ValueError: _description_

        Returns:
            TrxParquet: Instance of TrxParquet with data stored according to loadtype
        """
        header = TrxHeader.from_parquet_metadata(
            pq.read_schema(src).metadata.copy()
        )

        if loadtype == "lazy":
            # polars currently chokes on metadata in the schema
            # https://github.com/pola-rs/polars/issues/5117
            data = pl.scan_pyarrow_dataset(ds.dataset(src))
        elif loadtype == "memory_map":
            data = pl.read_parquet(
                source=str(src), memory_map=True, use_pyarrow=True
            )
        elif loadtype == "memory":
            data = pl.read_parquet(source=str(src), use_pyarrow=True)
        else:
            msg = "Unable to handle loadtype"
            raise ValueError(msg)

        if streamlines:
            data = data.filter(
                pl.col("protected_streamline").is_in(streamlines)
            )

        return cls(header=header, data=data)

    def to_file(self, dst: os.PathLike) -> os.PathLike:
        """Write TrxParquet to Parquet.

        Args:
            dst : Path of file to create

        Notes:
            When data are stored lazily, this reads them in.

        Returns:
            pathlib.Path : Absolute Path of created file
        """

        # Given that polars cannot handle metadata, the data attribute
        # is converted to pyarrow. That should be zero-copy, but
        # it cannot be done on a lazyframe. This means that lazyframes
        # are pulled into memory before write (instead of simply being
        # written to a file wth sink_parquet).
        if isinstance(self.data, pl.DataFrame):
            out = self.data.to_arrow().replace_schema_metadata(
                self.header.to_bytes_dict()
            )
        else:
            out = (
                self.data.collect()
                .to_arrow()
                .replace_schema_metadata(self.header.to_bytes_dict())
            )
        pq.write_table(out, where=dst, write_statistics=True)
        return Path(dst).absolute()

    def get_dtype_dict(self):
        """Get the dtype dictionary for the TrxParquet

        Returns
            A dictionary containing the schemas for each data element
        """

        return self.data.schema

    def set_index(self, indices: Collection[int]) -> "TrxParquet":
        """Update Indices used to represent streamlines. Useful during concatenation.

        Args:
            indices: New indices for the streamlines

        Returns:
            TrxParquet: self with updated indices.
        """
        if not len(indices) == self.n_streamlines:
            msg = "Number of new indices not match number of streamlines."
            raise ValueError(msg)
        old = self.data.select("protected_streamline").unique(
            maintain_order=True
        )
        if isinstance(old, pl.LazyFrame):
            old = old.collect()
        mapper = dict(zip(old.to_series(), indices))
        self.data = self.data.with_columns(
            protected_streamline=pl.col("protected_streamline").map_dict(
                mapper
            )
        )
        return self

    def to_memory(self) -> "TrxParquet":
        """Convert a Lazy TrxParquet to a RAM representation

        Args:
            resize -- Resize TrxParquet when converting to RAM representation

        Returns:
            A non memory mapped TrxParquet
        """
        if isinstance(self.data, pl.LazyFrame):
            self.data = self.data.collect()

        return self

    def concatenate(
        self,
        trx_list: list["TrxParquet"],
        delete_dpv: bool = False,
        delete_dps: bool = False,
        delete_groups: bool | None = None,
        check_space_attributes: bool = True,
    ) -> "TrxParquet":
        """Concatenate multiple TrxParquet together, support preallocation

        Args:
            trx_list -- A list containing TrxParquets to concatenate
            delete_dpv -- Delete dpv keys that do not exist in all the provided
                TrxParquets
            delete_dps -- Delete dps keys that do not exist in all the provided
                TrxParquet
            delete_groups -- Delete all the groups that currently exist in the
                TrxParquets
            check_space_attributes -- Verify that dimensions and size of data are
                similar between all the TrxParquets

        Notes:
            No effort is made to join on streamline index. If streamlines have
                matching indices, this may cause apparent elongation of streamlines.
                If this is not desired, then consider updating the indices of
                the elements in trx_list.

        Returns:
            TrxParquet representing the concatenated data

        """

        trx_list = [trx for trx in trx_list if trx.n_streamlines > 0]
        if len(trx_list) == 0:
            msg = "Inputs of concatenation were empty."
            raise ValueError(msg)

        if check_space_attributes:
            for curr_trx in trx_list:
                if not np.allclose(
                    self.header.VOXEL_TO_RASMM,
                    curr_trx.header.VOXEL_TO_RASMM,
                ) or not np.array_equal(
                    self.header.DIMENSIONS, curr_trx.header.DIMENSIONS
                ):
                    msg = "Wrong space attributes."
                    raise ValueError(msg)

        selection = cs.all()
        if delete_dpv:
            selection -= cs.starts_with("dpv")
        if delete_dps:
            selection -= cs.starts_with("dps")
        if delete_groups is not None:
            raise NotImplementedError

        trx_list.insert(0, self)
        return TrxParquet(
            data=pl.concat([trx.data.select(selection) for trx in trx_list]),  # type: ignore
            header=self.header,
        )

    def to_stf(self) -> stateful_tractogram.StatefulTractogram:
        """Convert to StatefulTractogram.

        Notes:
            This function will result in duplication of any dpv_ columns.

        Returns:
            stateful_tractogram.StatefulTractogram: dipy representation
                of object. Note that the
        """
        reference = (
            self.header.VOXEL_TO_RASMM,
            self.header.DIMENSIONS,
            affines.voxel_sizes(self.header.VOXEL_TO_RASMM),
            "".join(nb.orientations.aff2axcodes(self.header.VOXEL_TO_RASMM)),
        )
        return stateful_tractogram.StatefulTractogram(
            streamlines=self.streamlines,
            reference=reference,
            space=stateful_tractogram.Space.RASMM,
            data_per_point=self.data_per_vertex,
            data_per_streamline=self.data_per_streamline,
        )
