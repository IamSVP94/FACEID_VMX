import time
import faiss
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


# METRIC_L2  - Euclid
# METRIC_INNER_PRODUCT - cosine

def time_of_function(function):  # decorator for timing
    def wrapped(*args, **kwargs):
        start_time = time.perf_counter_ns()
        res = function(*args, **kwargs)
        # print(time.perf_counter_ns() - start_time)  # in nanosecs
        print((time.perf_counter_ns() - start_time) / 10 ** 9)  # in secs
        return res

    return wrapped


class Indexer:
    def __init__(self,
                 labels, embs,
                 normalize: bool = True,
                 n_clasters_index: int = 100,
                 device: str = 'cuda'):
        self.labels = labels
        self.embs = embs
        self.normalize = normalize
        if self.normalize:
            faiss.normalize_L2(self.embs)  # need for cosine metric
        self.n_clasters_index = n_clasters_index
        self.device = device
        self.index = self._get_index(self.embs.shape[-1], self.n_clasters_index)
        self.func_name = None  # only for print

    def __len__(self):
        return len(self.labels)

    def _get_index(self, dim, n_clasters):
        return None

    # @time_of_function
    def search(self, faces, nprobe=1, k_nearest=1):
        if nprobe > 1:
            self.index.nprobe = nprobe  # the number of nearby cells to search
        faces = np.squeeze(np.array(faces)).astype('float32')
        if self.normalize:
            faiss.normalize_L2(faces)  # need for cosine metric

        start_time = time.perf_counter_ns()
        Dists, Indexes = self.index.search(faces, k_nearest)  # actual search
        duration = time.perf_counter_ns() - start_time
        print(f'duration = {np.sum(duration) / 10 ** 9}')

        Labels = [self.labels[i[0]] for i in Indexes]
        return Labels, [d for d in np.squeeze(Dists).astype('float16')]


class Indexer_cdist(Indexer):
    def __init__(self, *args, **kwargs):
        super(Indexer_cdist, self).__init__(*args, **kwargs)
        self.func_name = 'Cos'

    # @time_of_function
    def search(self, faces, nprobe=1, k_nearest=1):
        Labels, Dists = [], []
        duration = 0
        for face in faces:
            if self.normalize:
                faiss.normalize_L2(face)  # need for cosine metric
            dists = []
            for person_emb in self.embs:
                person_emb = np.expand_dims(person_emb, axis=0)
                start_time = time.perf_counter_ns()
                dist = cdist(face, person_emb, metric='cosine')
                duration += time.perf_counter_ns() - start_time
                dists.append(dist)
            who = np.argmin(dists)
            dist_res = round(1 - dists[who][0][0], 4)
            Labels.append(self.labels[who])
            Dists.append(dist_res)
        print(f'duration = {duration / 10 ** 9}')
        return Labels, Dists


class Indexer_Float(Indexer):
    def __init__(self, *args, **kwargs):
        super(Indexer_Float, self).__init__(*args, **kwargs)
        self.func_name = 'Float'

    def _get_index(self, dim, n_clasters):
        index = faiss.IndexFlatIP(dim)  # build the index cosine
        index.add(self.embs)  # add vectors to the index
        return index


class Indexer_IVF(Indexer):
    def __init__(self, *args, **kwargs):
        super(Indexer_IVF, self).__init__(*args, **kwargs)
        self.func_name = 'IVF'

    def _get_index(self, dim, n_clasters):
        quantizer = faiss.IndexFlatIP(dim)  # build the index cosine
        index = faiss.IndexIVFFlat(quantizer, dim, n_clasters, faiss.METRIC_INNER_PRODUCT)

        if self.device == 'cuda':
            # gpu = faiss.StandardGpuResources()  # use a single GPU
            # index = faiss.index_cpu_to_gpu(gpu, 0, index)

            # ngpus = faiss.get_num_gpus()
            index = faiss.index_cpu_to_all_gpus(index)

        index.train(self.embs)
        index.add(self.embs)  # add vectors to the index
        return index


class Indexer_IVFPQ(Indexer):
    def __init__(self, *args, **kwargs):
        super(Indexer_IVFPQ, self).__init__(*args, **kwargs)
        self.func_name = 'IVFPQ'

    def _get_index(self, dim, n_clasters):
        quantizer = faiss.IndexFlatIP(dim)  # build the index cosine
        m = 512  # number of centroid IDs in final compressed vectors
        bits = 8  # number of bits in each centroid
        index = faiss.IndexIVFPQ(quantizer, dim, n_clasters, m, bits, faiss.METRIC_INNER_PRODUCT)

        if self.device == 'cuda':
            # gpu = faiss.StandardGpuResources()  # use a single GPU
            # index = faiss.index_cpu_to_gpu(gpu, 0, index)

            # ngpus = faiss.get_num_gpus()
            index = faiss.index_cpu_to_all_gpus(index)

        index.train(self.embs)
        index.add(self.embs)  # add vectors to the index
        return index


np.random.seed(2)  # make reproducible
faces_n = 10
faces = [np.random.random((1, 512)).astype('float32') for i in range(faces_n)]

etalons_n = 1000000
etalon_labels = [f'p_{i}' for i in range(etalons_n)]
etalons = np.random.random((etalons_n, 512)).astype('float32')

normalize = True  # True
n_clasters_index = int(etalons_n ** 0.5 * 16)  # 16000 for 1mln
nprobe = 2000  # 10

print(f'nprobe={nprobe}')
print(f'n_clasters_index={n_clasters_index}')
print()

# indexes_list = [Indexer_cdist, Indexer_Float, Indexer_IVF, Indexer_IVFPQ]
# indexes_list = [Indexer_cdist, Indexer_Float, Indexer_IVF]
indexes_list = [Indexer_Float, Indexer_IVF]

results_list_labels = list()
for idx, index_class in enumerate(indexes_list):
    index = index_class(labels=etalon_labels, embs=etalons,
                        normalize=normalize, n_clasters_index=n_clasters_index,
                        device='cuda',
                        )
    print(f'n etalons = {len(index)}, n faces = {len(faces)}')
    labels, dists = index.search(faces, nprobe=nprobe)
    results_list_labels.append(labels)
    print(f'"{index.func_name}"', labels, dists,
          sep='\n', end='\n' + '-' * 100 + '\n' * 2)
else:
    cos_res = results_list_labels[0]
    for idx, res in enumerate(results_list_labels[1:]):
        assert cos_res == res, f'{idx + 1}'
    print('complete! All is good!')
