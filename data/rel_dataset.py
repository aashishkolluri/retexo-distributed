import os
import hashlib

from relbench.datasets import get_dataset
from relbench.datasets.stackex import StackExDataset
from relbench.data import Database, Dataset, Table
import pandas as pd
import numpy as np
import shutil
import tempfile
import time

from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

class DistrRelBenchDataset(Dataset):
    name: str
    train_start_timestamp: Optional[pd.Timestamp] = None
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp
    # task_cls_list: List[Type[BaseTask]]

    db_dir: str = "db"
    
    def __init__(
        self,
        *,
        partition_dir: str = "data",
        part_id: int = 0,
        distributed: bool = False,
    ):
        if distributed:
            db_path = os.path.join(partition_dir, self.name, f"shard_{part_id}")
            print(f"loading Database object from {db_path}...")
            tic = time.time()
            db = Database.load(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")
        else:
            print("making Database object from raw files...")
            tic = time.time()
            db = self.make_db()
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            print("reindexing pkeys and fkeys...")
            tic = time.time()
            db.reindex_pkeys_and_fkeys()
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")
        
        super().__init__(
            db,
            self.train_start_timestamp,
            self.val_timestamp,
            self.test_timestamp,
            self.max_eval_time_frames,
            self.task_cls_list,
        )
    
    def make_db(self) -> Database:
        raise NotImplementedError
    
    def shardDataset(self, num_shards: int, folder: Union[str, os.PathLike]):
        raise NotImplementedError
    
    
    def pack_db(self, root: Union[str, os.PathLike]) -> Tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / self.db_dir
            print(f"saving Database object to {db_path}...")
            tic = time.time()
            self._full_db.save(db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

            print("making zip archive for db...")
            tic = time.time()
            zip_path = Path(root) / self.name / self.db_dir
            zip_path = shutil.make_archive(zip_path, "zip", db_path)
            toc = time.time()
            print(f"done in {toc - tic:.2f} seconds.")

        with open(zip_path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()

        print(f"upload: {zip_path}")
        print(f"sha256: {sha256}")

        return f"{self.name}/{self.db_dir}.zip", sha256

    
    # TODO add asserts throughout the process, especially on sizes
    def retrieve_foreign_rows(self, foreign_shards, current_shards, remaining_rows, allRows, OnKey="Id", foreignKey="Id"):
        for i in range(len(foreign_shards)):
            foreign_shard = foreign_shards[i]
            user_df_col = foreign_shard[[OnKey]]
            filtered_rows = allRows.merge(user_df_col, left_on=foreignKey, right_on=OnKey, how="inner")
            
            if OnKey == "Id":
                filtered_rows.rename(columns={'Id_x': 'Id'}, inplace=True)
                filtered_rows.drop(columns='Id_y', inplace=True)
            else:
                filtered_rows.drop(columns=OnKey, inplace=True)
            
            remaining_rows = remaining_rows[~remaining_rows["Id"].isin(filtered_rows["Id"])]
            if len(current_shards) > i:
                current_shards[i] = pd.concat([current_shards[i], filtered_rows])
            else:
                current_shards.append(filtered_rows)
                
            current_shards[i].drop_duplicates(subset='Id', keep='first', inplace=True)
        return remaining_rows



    def fill_shards(self, num_shards, table, shards, remainings):
        expected_batch_size = table.df.shape[0] / num_shards
        result = []
        for shard in shards:
            empty_slots = expected_batch_size - shard.shape[0]
            if empty_slots > 0:
                additional_slots = remainings.sample(n=min(int(empty_slots), int(remainings.shape[0])))
                shard = pd.concat([shard, additional_slots])
                remainings = remainings.drop(additional_slots.index)
            result.append(shard)
        return result
        
