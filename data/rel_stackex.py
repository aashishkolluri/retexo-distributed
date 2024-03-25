import os

from relbench.datasets import get_dataset
from relbench.datasets.stackex import StackExDataset
from relbench.data import Database, RelBenchDataset, Table
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List, Optional, Tuple, Type, Union
from data.rel_dataset import DistrRelBenchDataset


class DistrStackExDataset(DistrRelBenchDataset):
    name = "rel-stackex"
    # 2 years gap
    val_timestamp = pd.Timestamp("2019-01-01")
    test_timestamp = pd.Timestamp("2021-01-01")
    max_eval_time_frames = 1

    def __init__(
        self,
        *,
        part_id: int = "0",
        partition_dir: str = "data",
        distributed: bool = False,
    ):
        self.name=f"{self.name}"
        super().__init__(partition_dir=partition_dir, part_id=part_id, distributed=distributed)
        
    def fromShard(partition_dir: str, part_id: int):
        raise NotImplementedError
    
    
    def make_db(self) -> Database:
        r"""Process the raw files into a database."""
        url = "https://relbench.stanford.edu/data/relbench-forum-raw.zip"
        path = pooch.retrieve(
            url,
            known_hash="ad3bf96f35146d50ef48fa198921685936c49b95c6b67a8a47de53e90036745f",
            progressbar=True,
            processor=unzip_processor,
        )
        path = os.path.join(path, "raw")
        users = pd.read_csv(os.path.join(path, "Users.csv"))
        comments = pd.read_csv(os.path.join(path, "Comments.csv"))
        posts = pd.read_csv(os.path.join(path, "Posts.csv"))
        votes = pd.read_csv(os.path.join(path, "Votes.csv"))
        postLinks = pd.read_csv(os.path.join(path, "PostLinks.csv"))
        badges = pd.read_csv(os.path.join(path, "Badges.csv"))
        postHistory = pd.read_csv(os.path.join(path, "PostHistory.csv"))

        # tags = pd.read_csv(os.path.join(path, "Tags.csv")) we remove tag table here since after removing time leakage columns, all information are kept in the posts tags columns

        ## remove time leakage columns
        users.drop(
            columns=["Reputation", "Views", "UpVotes", "DownVotes", "LastAccessDate"],
            inplace=True,
        )

        posts.drop(
            columns=[
                "ViewCount",
                "AnswerCount",
                "CommentCount",
                "FavoriteCount",
                "CommunityOwnedDate",
                "ClosedDate",
                "LastEditDate",
                "LastActivityDate",
                "Score",
                "LastEditorDisplayName",
                "LastEditorUserId",
            ],
            inplace=True,
        )

        comments.drop(columns=["Score"], inplace=True)
        votes.drop(columns=["BountyAmount"], inplace=True)

        ## change time column to pd timestamp series
        comments["CreationDate"] = pd.to_datetime(comments["CreationDate"])
        badges["Date"] = pd.to_datetime(badges["Date"])
        postLinks["CreationDate"] = pd.to_datetime(postLinks["CreationDate"])

        postHistory["CreationDate"] = pd.to_datetime(postHistory["CreationDate"])
        votes["CreationDate"] = pd.to_datetime(votes["CreationDate"])
        posts["CreationDate"] = pd.to_datetime(posts["CreationDate"])
        users["CreationDate"] = pd.to_datetime(users["CreationDate"])

        tables = {}

        tables["comments"] = Table(
            df=pd.DataFrame(comments),
            fkey_col_to_pkey_table={
                "UserId": "users",
                "PostId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["badges"] = Table(
            df=pd.DataFrame(badges),
            fkey_col_to_pkey_table={
                "UserId": "users",
            },
            pkey_col="Id",
            time_col="Date",
        )

        tables["postLinks"] = Table(
            df=pd.DataFrame(postLinks),
            fkey_col_to_pkey_table={
                "PostId": "posts",
                "RelatedPostId": "posts",  ## is this allowed? two foreign keys into the same primary
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["postHistory"] = Table(
            df=pd.DataFrame(postHistory),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["votes"] = Table(
            df=pd.DataFrame(votes),
            fkey_col_to_pkey_table={"PostId": "posts", "UserId": "users"},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["users"] = Table(
            df=pd.DataFrame(users),
            fkey_col_to_pkey_table={},
            pkey_col="Id",
            time_col="CreationDate",
        )

        tables["posts"] = Table(
            df=pd.DataFrame(posts),
            fkey_col_to_pkey_table={
                "OwnerUserId": "users",
                "ParentId": "posts",  # notice the self-reference
                "AcceptedAnswerId": "posts",
            },
            pkey_col="Id",
            time_col="CreationDate",
        )

        return Database(tables)
    
    
    def shardDataset(self, num_shards: int = 2, folder: Union[str, os.PathLike] = "data") -> List[Database]: 
        r"""Shard the dataset horizontally to simulate distributed data.
        
        Datasets are stored in the cache."""
        
        db = self.db
        
        shards = []
        
        # Users table
        users = db.table_dict["users"]
        user_shards = np.array_split(users.df, num_shards)   
        for i in range(len(user_shards)):
            user_shards[i]["part_id"] = i
        shards.append(
                {"name": "users", 
                    "df": user_shards, 
                    "fk_to_table": users.fkey_col_to_pkey_table,
                    "time_col": users.time_col
                }
        )
        
        # Votes table
        votes = db.table_dict["votes"]  
        vote_shards = []
        remaining_votes = votes.df
        
        remaining_posts = self.retrieve_foreign_rows(shards[0]["df"], vote_shards, remaining_votes, votes.df, foreignKey="UserId")
        # Filling the shards with the remaining votes 
        vote_shards = self.fill_shards(num_shards, votes, vote_shards, remaining_votes)            
        shards.append(
                {"name": "votes", 
                    "df": vote_shards, 
                    "fk_to_table": votes.fkey_col_to_pkey_table,
                    "time_col": votes.time_col
                }
                )   
        
        # Posts table
        posts = db.table_dict["posts"]
        post_shards = []
        remaining_posts = posts.df
            
        remaining_posts = self.retrieve_foreign_rows(shards[0]["df"], post_shards, remaining_posts, posts.df, foreignKey="OwnerUserId")
        remaining_posts = self.retrieve_foreign_rows(shards[1]["df"], post_shards, remaining_posts, posts.df, OnKey="PostId", foreignKey="Id")
            
        post_shards = self.fill_shards(num_shards, posts, post_shards, remaining_posts)            
        shards.append(
        {"name": "posts", 
            "df": post_shards, 
            "fk_to_table": posts.fkey_col_to_pkey_table,
            "time_col": posts.time_col
        }
        )   
        
        
        # Badges Table
        badges = db.table_dict["badges"]
        badge_shards = []
        remaining_badges = badges.df
            
        remaining_badges = self.retrieve_foreign_rows(shards[0]["df"], badge_shards, remaining_badges, badges.df, foreignKey="UserId")

        badge_shards = self.fill_shards(num_shards, badges, badge_shards, remaining_badges)            
        shards.append(
        {"name": "badges", 
            "df": badge_shards, 
            "fk_to_table": badges.fkey_col_to_pkey_table,
            "time_col": badges.time_col
        }
        ) 
        
        # Comments Table
        comments = db.table_dict["comments"]
        comment_shards = []
        remaining_comments = comments.df
        
        remaining_comments = self.retrieve_foreign_rows(shards[0]["df"], comment_shards, remaining_comments, comments.df, foreignKey="UserId")
        remaining_comments = self.retrieve_foreign_rows(shards[2]["df"], comment_shards, remaining_comments, comments.df, foreignKey="PostId")
        
        comment_shards = self.fill_shards(num_shards, comments, comment_shards, remaining_comments)
        shards.append(
        {"name": "comments", 
            "df": comment_shards, 
            "fk_to_table": comments.fkey_col_to_pkey_table,
            "time_col": comments.time_col
        }
        )
        
        # PostHistory Table
        postHistory = db.table_dict["postHistory"]
        postHistory_shards = []
        remaining_postHistory = postHistory.df
        
        remaining_postHistory = self.retrieve_foreign_rows(shards[0]["df"], postHistory_shards, remaining_postHistory, postHistory.df, foreignKey="UserId")
        remaining_postHistory = self.retrieve_foreign_rows(shards[2]["df"], postHistory_shards, remaining_postHistory, postHistory.df, foreignKey="PostId")
        
        postHistory_shards = self.fill_shards(num_shards, postHistory, postHistory_shards, remaining_postHistory)
        shards.append(
        {"name": "postHistory", 
            "df": postHistory_shards, 
            "fk_to_table": postHistory.fkey_col_to_pkey_table,
            "time_col": postHistory.time_col
        }
        )
        
        # PostLinks Table
        postLinks = db.table_dict["postLinks"]
        postLinks_shards = []
        remaining_postLinks = postLinks.df
        
        remaining_postLinks = self.retrieve_foreign_rows(shards[2]["df"], postLinks_shards, remaining_postLinks, postLinks.df, foreignKey="PostId")
        remaining_postLinks = self.retrieve_foreign_rows(shards[2]["df"], postLinks_shards, remaining_postLinks, postLinks.df, foreignKey="RelatedPostId")
        
        postLinks_shards = self.fill_shards(num_shards, postLinks, postLinks_shards, remaining_postLinks)
        shards.append(
        {"name": "postLinks", 
            "df": postLinks_shards, 
            "fk_to_table": postLinks.fkey_col_to_pkey_table,
            "time_col": postLinks.time_col}
        )
        
        for shard in shards:
            print("size of shard table " + shard["name"])
            total = db.table_dict[shard["name"]].df.shape[0]
            sum = 0
            for i in range(num_shards):
                print("Shard " + str(i) + " is " + str(shard["df"][i].shape[0]))
                sum += shard["df"][i].shape[0]
                
            duplicated = sum - total
            print("Portion of data duplicated "+ str((duplicated/total*100)) + "%")
        
        
        # Make a database for each shard
        databases = []
        for i in range(num_shards): 
            tables = {}
            for shard in shards:
                tables[shard["name"]] = Table(
                    df=shard["df"][i],
                    fkey_col_to_pkey_table=shard["fk_to_table"],
                    pkey_col="Id",
                    time_col=shard["time_col"],
                )
                
            db = Database(tables)    
            # save db to specific folder
            db.save(f"{folder}/{self.name}/shard_{i}")
            databases.append(db)
            
        return databases
    
    
    
    
# main function to debug the shardDataset function
if __name__ == "__main__":
    dataset = DistrStackExDataset(process=True)
    # for i in range(2, 11):
    #     print("FOR i = "+ str(i) + " SHARDS" )
    path = os.path.join(os.getcwd(), "data")
    dataset.shardDataset(num_shards=4, folder=path)