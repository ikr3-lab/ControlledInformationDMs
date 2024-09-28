from bs4 import BeautifulSoup
import os
import pandas as pd


def parse_topics(filepath: str):
    with open(filepath) as f:
        soup = BeautifulSoup(f, "lxml")
        topics = soup.find_all("top")
        topics_dict = {}
        for topic in topics:
            id = topic.find("num").text[9:].strip()
            query = topic.find("title").text.strip()
            topics_dict[id] = query

    topics_df = pd.DataFrame([topics_dict]).T.reset_index()
    topics_df.columns = ['qid', 'query']
    topics_df['qid'] = topics_df['qid'].astype('str')
    topics_df['query'] = topics_df['query'].astype('str').replace("\"", "")
    topics_df['query'] = topics_df['query'].str.replace('"', '')
    return topics_df


def parse_qrels(filepath: str, docs):
    with open(filepath) as f:
        qrels_df = pd.read_csv(filepath, sep=" ", header=None)
        qrels_df.columns = ["qid", "iteration", "docno", "label"]
        qrels_df.drop("iteration", axis=1, inplace=True)
        qrels_df['qid'] = qrels_df['qid'].astype('str')
        qrels_df= qrels_df[qrels_df['docno'].isin([doc.id for doc in docs])]
        return qrels_df
