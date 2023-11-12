import os
import time

import polars as pl
from trx import fetcher

from trx_polars import trxparquet


def test_example():
    """Test that example trxparquets can be created from scratch"""
    try:
        trxparquet.init_example_trxparquet(2, 2)
        basic = True
    except BaseException:
        basic = False

    try:
        trxparquet.init_example_trxparquet(2, 2, 2)
        with_per_streamline = True
    except BaseException:
        with_per_streamline = False

    try:
        trxparquet.init_example_trxparquet(2, 2, 2, 2)
        with_per_vertex = True
    except BaseException:
        with_per_vertex = False

    assert all([basic, with_per_streamline, with_per_vertex])


def test_load_trxfile(tmp_path):
    """Test that trxparquets can be loaded from trxfiles"""
    os.environ["TRX_HOME"] = str(tmp_path)
    data = dict(
        (k, v)
        for k, v in fetcher.get_testing_files_dict().items()
        if k == "gold_standard.zip"
    )
    fetcher.fetch_data(data)
    del os.environ["TRX_HOME"]

    try:
        trxparquet.TrxParquet.from_trx_file(
            tmp_path / "gold_standard" / "gs_fldr.trx"
        )
        loaded = True
    except BaseException:
        loaded = False

    assert loaded


def test_concat():
    """Test that trxparquets can be concatenated"""
    a = trxparquet.init_example_trxparquet(2, 2)
    b = trxparquet.init_example_trxparquet(2, 2)

    assert pl.concat([a.data, b.data]).frame_equal(a.concatenate([b]).data)  # type: ignore


def test_load_large(tmp_path):
    """Test that loading large files is quick"""
    tmp_file = tmp_path / "trx.parquet"
    src = trxparquet.init_example_trxparquet(
        n_streamlines=100000, n_points_per_streamline=100
    ).to_file(tmp_file)

    start = time.time()
    trxparquet.TrxParquet.from_file(src)

    assert time.time() - start < 2


def test_to_stateful():
    """Test that trxparquets can be converted to stateful tractograms"""
    try:
        trxparquet.init_example_trxparquet(2, 2).to_stf()
        basic = True
    except BaseException:
        basic = False

    try:
        trxparquet.init_example_trxparquet(2, 2, 2).to_stf()
        with_per_streamline = True
    except BaseException:
        with_per_streamline = False

    try:
        trxparquet.init_example_trxparquet(2, 2, 2, 2).to_stf()
        with_per_vertex = True
    except BaseException:
        with_per_vertex = False

    assert all([basic, with_per_streamline, with_per_vertex])
