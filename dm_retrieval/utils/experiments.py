from dm_retrieval.utils.doc import DocList
import pyterrier as pt
import os
import mlflow
import statistics
import matplotlib.pyplot as plt
import numpy as np


def figure(privacy_list, list2, mode="map"):
    fig, ax = plt.subplots()
    ax.plot(privacy_list, list2, marker='o')
    ax.set_xlabel('Privacy')
    if mode == "map":
        ax.set_ylabel('MAP')
        ax.set_title('MAP / Privacy')
    else:
        ax.set_ylabel('NDCG')
        ax.set_title('NDCG / Privacy')
    ax.set_xlim(0.5, 1)
    ax.set_ylim(0, 0.6)
    ax.set_xticks(np.arange(0.5, 1, 0.05))
    ax.set_yticks(np.arange(0, 0.6, 0.05))
    ax.grid()
    return fig


def calculate_privacy_metrics(docs: DocList, privacy_list: list):
    total_ents = 41816
    inferred_sum = 0
    mask_sum = 0
    for doc in docs:
        mask_sum += doc.mask_count
        inferred_sum += doc.inferred_count
    privacy = 1 - (inferred_sum / total_ents)
    privacy_list.append(privacy)
    return inferred_sum, mask_sum, privacy


def initialize_result_objects(wmodels: list):
    privacy_list = []
    map_dict = {}
    ndcg_dict = {}

    for model in wmodels:
        map_dict[model] = []
        ndcg_dict[model] = []
    return privacy_list, map_dict, ndcg_dict


def indexing(indexes_folder: str, docs: DocList, experiment_name: str = None, experiment_id: str = None):
    if not experiment_id:
        experiment_id = mlflow.create_experiment(experiment_name)
    index_path = os.path.join(indexes_folder, experiment_id)
    docs_indexable = [doc.to_index_doc() for doc in docs]
    iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 40, 'text': 4096})
    iter_indexer.index(docs_indexable)
    return (pt.IndexFactory.of(os.path.join(index_path, 'data.properties')), experiment_id)


def experiment(wmodels: list, metrics: list, experiment_id: str, index, topics, qrels, privacy, mask_sum, inferred_sum,
               map_dict, ndcg_dict, query_expansion: bool = False, query_expansion_fun=None):
    model_dict = {}
    for wmodel in wmodels:
        RUN_NAME = wmodel
        with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
            model = pt.BatchRetrieve(index, wmodel=wmodel)
            mlflow.log_param("model", wmodel)
            if query_expansion:
                query_expansion_log = query_expansion_fun.__name__.rsplit(".")[0]
                qe = query_expansion_fun(index)
                model_exp = [model >> qe >> model]
            else:
                query_expansion_log = "False"
                model_exp = [model]
            mlflow.log_param("query_expansion", query_expansion_log)
            metric_df = pt.Experiment(model_exp, topics, qrels, eval_metrics=metrics)
            for metric in metrics:
                mlflow.log_metrics(metric_df.drop('name', axis=1).iloc[0].to_dict())
            model_dict[wmodel] = metric_df.drop('name', axis=1).iloc[0].to_dict()
            mlflow.log_metric('Privacy ', privacy)
            mlflow.log_metric('Mask sum', mask_sum)
            mlflow.log_metric('Inferred sum', inferred_sum)

    for key in model_dict:
        map_dict[key].append(model_dict[key]["map"])
        ndcg_dict[key].append(model_dict[key]["ndcg"])


def experiment_results(experiment_name: str, wmodels: list, privacy_list: list, map_dict: dict, ndcg_dict: dict):
    experiment_id = mlflow.create_experiment(experiment_name)
    for wmodel in wmodels:
        RUN_NAME = wmodel
        with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
            mlflow.log_param("model", wmodel)
            mlflow.log_metric('Avg Privacy ', statistics.mean(privacy_list))
            mlflow.log_metric('Avg MAP', statistics.mean(map_dict[wmodel]))
            mlflow.log_metric('Avg ndcg', statistics.mean(ndcg_dict[wmodel]))
            mlflow.log_param('Privacy ', privacy_list)
            mlflow.log_param('MAP', map_dict[wmodel])
            mlflow.log_param('NDCG', ndcg_dict[wmodel])
            mlflow.log_figure(figure(privacy_list, map_dict[wmodel], mode="map"), "mlflow/map_privacy.png")
            mlflow.log_figure(figure(privacy_list, ndcg_dict[wmodel], mode="ndcg"), "mlflow/ndcg_privacy.png")


##########
def initialize_result_objects_qe(wmodels: list, qe_fun_list):
    privacy_list = []
    map_dict = {}
    ndcg_dict = {}
    for model in wmodels:
        map_dict[model] = {}
        ndcg_dict[model] = {}

    for model in wmodels:
        for qe in qe_fun_list:
            map_dict[model][qe.__name__] = []
            ndcg_dict[model][qe.__name__] = []
    return privacy_list, map_dict, ndcg_dict


def experiment_qe(wmodels: list, metrics: list, experiment_id: str, index, topics, qrels, privacy, mask_sum,
                  inferred_sum,
                  map_dict, ndcg_dict, qe_fun_list=None):
    model_dict = {}
    for model in wmodels:
        model_dict[model] = {}
        model_dict[model] = {}

    for wmodel in wmodels:
        for qe in qe_fun_list:
            print(qe)
            RUN_NAME = wmodel + ": " + qe.__name__
            with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
                model = pt.BatchRetrieve(index, wmodel=wmodel)
                mlflow.log_param("model", wmodel)
                query_expansion_log = qe.__name__.rsplit(".")[0]
                qe_step = qe(index)
                model_exp = [model >> qe_step >> model]
                mlflow.log_param("query_expansion", query_expansion_log)
                metric_df = pt.Experiment(model_exp, topics, qrels, eval_metrics=metrics)
                for metric in metrics:
                    mlflow.log_metrics(metric_df.drop('name', axis=1).iloc[0].to_dict())
                model_dict[wmodel][qe.__name__] = metric_df.drop('name', axis=1).iloc[0].to_dict()
                mlflow.log_metric('Privacy ', privacy)
                mlflow.log_metric('Mask sum', mask_sum)
                mlflow.log_metric('Inferred sum', inferred_sum)

    for key in model_dict:
        for qe in model_dict[key]:
            map_dict[key][qe].append(model_dict[key][qe]["map"])
            ndcg_dict[key][qe].append(model_dict[key][qe]["ndcg"])


def experiment_results_qe(experiment_name: str, wmodels: list, privacy_list: list, map_dict: dict, ndcg_dict: dict,
                          qe_fun_list):
    experiment_id = mlflow.create_experiment(experiment_name)
    for wmodel in wmodels:
        for qe in qe_fun_list:
            RUN_NAME = wmodel + ": " + qe.__name__
            with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
                mlflow.log_param("model", wmodel)
                mlflow.log_metric('Avg Privacy ', statistics.mean(privacy_list))
                mlflow.log_metric('Avg MAP', statistics.mean(map_dict[wmodel][qe.__name__]))
                mlflow.log_metric('Avg ndcg', statistics.mean(ndcg_dict[wmodel][qe.__name__]))
                mlflow.log_param('Privacy ', privacy_list)
                mlflow.log_param('MAP', map_dict[wmodel][qe.__name__])
                mlflow.log_param('NDCG', ndcg_dict[wmodel][qe.__name__])
                mlflow.log_figure(figure(privacy_list, map_dict[wmodel][qe.__name__], mode="map"),
                                  "mlflow/map_privacy.png")
                mlflow.log_figure(figure(privacy_list, ndcg_dict[wmodel][qe.__name__], mode="ndcg"),
                                  "mlflow/ndcg_privacy.png")
