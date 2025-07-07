import random
import numpy as np
import torch
import os
from torch.utils.data import IterableDataset

class NPZReader(IterableDataset):
    """Dataset for loading NPZ files.

    Args:
        file_path (str): Path to NPZ data file.
        in_variables (list): List of input variables.
        out_variables (list): List of output variables.
        predictions (list, optional): List of predictions elements to include in output. 
            If provided must all be positive integers. Defaults to [] (current timestamp).
        history (list, optional): List of history elements to include in input. 
            If provided must all be negative integers. Defaults to [] (current timestamp).
        overflow_file_paths (list[str], optional): List of overflow NPZ data files to prepend. 
            Must be in order from oldest to newest. Will be concatenated as [overflow[0], ... overflow[-1], file_path]

    """
    def __init__(
        self,
        file_path: str,
        in_variables: list,
        out_variables: list,
        predictions: list = [],
        history: list = [],
        overflow_file_paths: list = [],
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.in_variables = in_variables
        self.out_variables = out_variables
        self.predictions = sorted(predictions)
        self.history = sorted(history)
        self.overflow_file_paths = overflow_file_paths
        if self.history:
            assert all(h < 0 and isinstance(h, int) for h in self.history), "All history elements must be negative integers"
        if self.predictions:
            assert all(p > 0 and isinstance(p, int) for p in self.predictions), "All prediction elements must be positive integers"
        self.history_range = min(self.history) * -1 if self.history else 0
        self.predict_range = max(self.predictions) if self.predictions else 0

    def __iter__(self):
        data = dict(np.load(self.file_path))
        if self.overflow_file_paths:
            curr_file, total_files = 1, len(self.overflow_file_paths)
            history_to_append = self.history_range
            while history_to_append > 0 and curr_file <= total_files:
                overflow = dict(np.load(self.overflow_file_paths[total_files - curr_file]))
                H = -history_to_append if len(overflow['dates']) >= history_to_append else 0
                history_to_append -= len(overflow['dates'])
                if (data['dates'][0] - overflow['dates'][-1]).astype('timedelta64[D]') > 1:
                        print(f"Warning: Gap between last element of overflow {overflow['dates'][-1]} and first element of data {data['dates'][0]} is greater than 1 day")
                for v in data.keys():
                    data[v] = np.concatenate([overflow[v][H:], data[v]])
                curr_file += 1
        
        if not torch.distributed.is_initialized():
            global_rank = 0
            world_size = 1
        else:
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        
         # split data across ranks. each rank will get an identical slice to avoid synchronization issues. if data cannot be split evenly, remainder will be discarded 
        timesteps_per_rank = (len(data['dates']) + ((world_size - 1)*(self.predict_range + self.history_range))) // world_size
        assert timesteps_per_rank > (self.predict_range + self.history_range), f"Data per rank with size {timesteps_per_rank} is not large enough for history timestamps {self.history} and predictions {self.predictions}. Decrease devices."
        rank_start_idx = global_rank * (timesteps_per_rank - (self.predict_range + self.history_range))
        rank_end_idx = rank_start_idx + timesteps_per_rank
        data_per_rank = {v: data[v][rank_start_idx:rank_end_idx] for v in data.keys()}

        #within each rank split data across workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # split data across workers. each worker will get an identical slice to avoid synchronization issues. if data cannot be split evenly, remainder will be discarded 
        timesteps_per_worker = (timesteps_per_rank + ((num_workers - 1)*(self.predict_range + self.history_range))) // num_workers
        assert timesteps_per_worker > (self.predict_range + self.history_range), f"Data per worker with size {timesteps_per_worker} is not large enough for history timestamps {self.history} and predictions {self.predictions}. Decrease num_workers."
        worker_start_idx = worker_id * (timesteps_per_worker - (self.predict_range + self.history_range))
        worker_end_idx = worker_start_idx + timesteps_per_worker
        data_per_worker = {v: data_per_rank[v][worker_start_idx:worker_end_idx] for v in data_per_rank.keys()}

        print(f'Rank: {global_rank + 1}/{world_size} gets {rank_start_idx} to {rank_end_idx}. Worker {worker_id + 1}/{num_workers} in rank {global_rank + 1} gets {worker_start_idx} to {worker_end_idx}')
        yield data_per_worker, self.in_variables, self.out_variables, self.predictions, self.history, self.predict_range, self.history_range


class Forecast(IterableDataset):
    def __init__(
        self, 
        dataset: NPZReader, 
        forecast_dir: str = None,
        load_forecast_members: bool = False,
        normalize_data: bool = False, 
        in_transforms = None, 
        out_transforms = None,
        filter_mjo_events: bool = False
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.forecast_dir = forecast_dir
        self.load_forecast_members = load_forecast_members
        self.normalize_data = normalize_data
        self.in_transforms = in_transforms
        self.out_transforms = out_transforms
        self.filter_mjo_events = filter_mjo_events
      
    def __iter__(self):
        for data, in_variables, out_variables, predictions, history, predict_range, history_range in self.dataset:
            
            for t in range(history_range, len(data['dates']) - predict_range):
                if not self.filter_mjo_events or (self.filter_mjo_events and data['amplitude'][t] > 1):
                    in_data = torch.stack([torch.tensor([data[v][t + h] for h in history + [0]], dtype=torch.get_default_dtype()) for v in in_variables], dim=1)
                    in_timestamps = np.array([data['dates'][t + h] for h in history + [0]])
                    
                    if predict_range == 0:
                        out_data = torch.stack([torch.tensor(data[v][t], dtype=torch.get_default_dtype()) for v in out_variables], dim=1)            
                        out_timestamps = np.array(data['dates'][t])
                    else:
                        out_data = torch.stack([torch.tensor([data[v][t + p] for p in predictions], dtype=torch.get_default_dtype()) for v in out_variables], dim=1)
                        out_timestamps = np.array([data['dates'][t + p] for p in predictions])

                    # if forecast_dir is provided, only load samples with future forecasts
                    if self.forecast_dir:
                        forecast_mean_file = f"{str(data['dates'][t]).split('T')[0]}_mean.npz"
                        if os.path.exists(os.path.join(self.forecast_dir, forecast_mean_file)):
                            forecast_npz_data = np.load(os.path.join(self.forecast_dir, forecast_mean_file))
                            forecast_data = torch.stack([torch.tensor(forecast_npz_data[v], dtype=torch.get_default_dtype()) for v in in_variables], dim=2)
                            forecast_timestamps = np.array(forecast_npz_data['dates'])
                            # assert len(forecast_timestamps) == len(out_timestamps), f'Found mismatch between forecast length {len(forecast_timestamps)} and predict length {len(out_timestamps)}'
                            if self.load_forecast_members:
                                forecast_members_file = f"{str(data['dates'][t]).split('T')[0]}_members.npz"
                                if os.path.exists(os.path.join(self.forecast_dir, forecast_members_file)):
                                    forecast_npz_data = np.load(os.path.join(self.forecast_dir, forecast_members_file))
                                    forecast_member_data = torch.stack([torch.tensor(forecast_npz_data[v], dtype=torch.get_default_dtype()) for v in in_variables], dim=2)
                                    forecast_data = torch.concatenate([forecast_data, forecast_member_data], dim=0)
                                    forecast_member_timestamps = np.array(forecast_npz_data['dates'])
                                    assert (forecast_timestamps == forecast_member_timestamps).all(), f"Found mismatch between forecast member timestamps and forecast mean timestamps for {str(data['dates'][t]).split('T')[0]}"     
                        else:
                            continue
                    
                    #remove missing datapoints
                    if torch.isnan(in_data).any() or torch.isnan(out_data).any():
                        continue

                    if self.normalize_data:
                        yield self.in_transforms.normalize(in_data), self.out_transforms.normalize(out_data), self.in_transforms.normalize(forecast_data) if self.forecast_dir else None, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps if self.forecast_dir else None
                    else:
                        yield in_data, out_data, forecast_data if self.forecast_dir else None, in_variables, out_variables, in_timestamps, out_timestamps, forecast_timestamps if self.forecast_dir else None


class ShuffleIterableDataset(IterableDataset):
    def __init__(
            self, 
            dataset, 
            max_buffer_size: int = 100
    ) -> None:
        super().__init__()
        assert max_buffer_size > 0, 'Buffer size must be > 0'
        self.dataset = dataset
        self.max_buffer_size = max_buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.max_buffer_size:
                idx = random.randint(0, self.max_buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()


def collate_fn(batch):
    batch = list(zip(*batch)) 
    in_data = torch.stack(batch[0])
    out_data = torch.stack(batch[1])
    forecast_data = torch.stack(batch[2]) if batch[2][0] is not None else None
    in_variables = batch[3][0]
    out_variables = batch[4][0]
    in_timestamps = np.array(batch[5])
    out_timestamps = np.array(batch[6])
    forecast_timestamps = np.array(batch[7]) if batch[7][0] is not None else None
    return (
        in_data,
        out_data,
        forecast_data,
        in_variables,
        out_variables,
        in_timestamps,
        out_timestamps,
        forecast_timestamps
    )