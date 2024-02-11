import os
import numpy as np
import torch
from torch import nn
from graph_generation import gen_multi_level_local_graph_v3
from Instseg_model import MultiLayerFastLocalGraphModelV2
import matplotlib.pyplot
from dataset import gtloader, pcloader, assign_weld_region, MyDataset
import time
import torch.utils.data as data_utils


graph_gen_kwargs = {
    'add_rnd3d': True,
    'base_voxel_size': 0.8,
    'downsample_method': 'random',
    'level_configs': [
        {'graph_gen_kwargs': {'num_neighbors': 64, 'radius': 0.4},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 0,
         'graph_scale': 1},
        {'graph_gen_kwargs': {'num_neighbors': 192, 'radius': 1.2},
         'graph_gen_method': 'disjointed_rnn_local_graph_v3',
         'graph_level': 1,
         'graph_scale': 1}]
}

class Detector():
    def __init__(self, cuda=False, model_params='./model/param_1212_updated_loss_06_64_192_49_1099'):
        a = time.time()
        self.model = MultiLayerFastLocalGraphModelV2(num_classes=3, max_instance_no=7)
        self.cuda = cuda
        if self.cuda and torch.cuda.is_available():
            print("GPU(s) enabled")
            self.model = self.model.cuda()

        self.model.load_state_dict(torch.load(model_params))
        self.predictions = {}
        self.labels = {}
        self.terminal_points = None
        b = time.time()
        print('Model Spawn Time: ', b-a)


    def process_sample(self, scan_path, label_path=None):
        a = time.time()
        res = {}
        if label_path:
            datasample = MyDataset(dataset=[[scan_path, label_path]])[0]

        else:
            pointxyz, offset = pcloader(scan_path)  ## possible remove offset so as not to rezero cloud points
            print(offset)
            # pointxyz += offset
            # print("removed offset")

            vertex_coord_list, keypoint_indices_list, edges_list = \
            gen_multi_level_local_graph_v3(pointxyz,0.6,graph_gen_kwargs['level_configs'])
            
            vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
            keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
            edges_list = [e.astype(np.int32) for e in edges_list]

            vertex_coord_list = [torch.from_numpy(item) for item in vertex_coord_list]
            keypoint_indices_list = [torch.from_numpy(item).long() for item in keypoint_indices_list]
            edges_list = [torch.from_numpy(item).long() for item in edges_list]

            datasample = (vertex_coord_list, keypoint_indices_list, edges_list, None, None)

        if self.cuda:
            cuda_datasample = []
            for item in datasample:
                if item:
                    if not isinstance(item, torch.Tensor):
                        item = [torch.squeeze(x,0).cuda() for x in item]
                    else:
                        item = torch.squeeze(item,0).cuda() 
                cuda_datasample += [item]
            datasample = cuda_datasample

        vertex_coord_list, keypoint_indices_list, edges_list, cls_labels, inst_labels = datasample
        
        # cls_preds, inst_preds = self.model(datasample, is_training=False)
        cls_preds, inst_preds = self.model(datasample)
        cls_preds = torch.argmax(cls_preds, dim=1)
        inst_preds = torch.argmax(inst_preds, dim=1)
        terminal_points = vertex_coord_list[2]
        
        if self.cuda:
            res['cls_preds'] = cls_preds.cpu().detach().numpy()
            res['inst_preds'] = inst_preds.cpu().detach().numpy()
            res['terminal_points'] = terminal_points.cpu().detach().numpy()
            res['cls_labels'] = cls_labels.cpu().detach().numpy() if cls_labels != None else None
            res['inst_labels'] = inst_labels.cpu().detach().numpy() if inst_labels != None else None
        else:
            res['cls_preds'] = cls_preds.detach().numpy()
            res['inst_preds'] = inst_preds.detach().numpy()
            res['terminal_points'] = terminal_points.detach().numpy()
            res['cls_labels'] = cls_labels.detach().numpy() if cls_labels != None else None
            res['inst_labels'] = inst_labels.detach().numpy() if inst_labels != None else None

        b = time.time()
        print("Inference Time: ", b-a)

        self.predictions['cls'] = res['cls_preds']
        self.predictions['inst'] = res['inst_preds']
        self.labels['cls'] = res['cls_labels']
        self.labels['inst'] = res['inst_labels']
        self.terminal_points = res['terminal_points']

        return res

    def extract_welds_from(self, cls_seg, inst_seg):
        if np.any(cls_seg) and np.any(inst_seg):
            unique_inst = np.unique(inst_seg)
            unique_cls = np.unique(cls_seg)

            weldsInst = {}
            weldsCls = {}
            welds = {}
            welds_by_instance = {}
            welds_by_type = {}

            for inst in unique_inst:
                if inst == 0:
                    continue
                m_inst = np.asarray(inst_seg == inst)
                weldsInst[inst] = m_inst
                welds_by_instance[inst] = self.terminal_points[weldsInst[inst].nonzero()[0]]

            for cls in unique_cls:
                if cls == 0:
                    continue
                m_cls = np.asarray(cls_seg == cls)
                weldsCls[cls] = m_cls
                welds_by_type[cls] = self.terminal_points[weldsCls[cls].nonzero()[0]]

            ## need to filter for both weld type and weld instance
            for inst in weldsInst.keys():
                splits = []
                splits_count = []
                
                for cls in weldsCls.keys():
                    splits.append(np.logical_and(weldsCls[cls], weldsInst[inst]).nonzero()[0])
                    splits_count.append(np.logical_and(weldsCls[cls], weldsInst[inst]).nonzero()[0].shape)
                
                welds[inst] = self.terminal_points[splits[splits_count.index(max(splits_count))]]
                
            
            return welds, welds_by_type, welds_by_instance

        else:
            print("A sample must be processed through the network before welds can be extracted")
            return None, None, None


    # def extract_welds(self):
    #     if np.any(self.predictions['cls']) and np.any(self.predictions['inst']):
    #         unique_inst = np.unique(self.predictions['inst'])
    #         unique_cls = np.unique(self.predictions['cls'])

    #         weldsInst = {}
    #         weldsCls = {}
    #         welds = {}
    #         self.welds_by_instance = {}
    #         self.welds_by_type = {}

    #         for inst in unique_inst:
    #             if inst == 0:
    #                 continue
    #             m_inst = np.asarray(self.predictions['inst'] == inst)
    #             weldsInst[inst] = m_inst
    #             self.welds_by_instance[inst] = self.terminal_points[weldsInst[inst].nonzero()[0]]

    #         for cls in unique_cls:
    #             if cls == 0:
    #                 continue
    #             m_cls = np.asarray(self.predictions['cls'] == cls)
    #             weldsCls[cls] = m_cls
    #             self.welds_by_type[cls] = self.terminal_points[weldsCls[cls].nonzero()[0]]

    #         ## need to filter for both weld type and weld instance
    #         for inst in weldsInst.keys():
    #             splits = []
    #             splits_count = []
                
    #             for cls in weldsCls.keys():
    #                 splits.append(np.logical_and(weldsCls[cls], weldsInst[inst]).nonzero()[0])
    #                 splits_count.append(np.logical_and(weldsCls[cls], weldsInst[inst]).nonzero()[0].shape)
                
    #             welds[inst] = self.terminal_points[splits[splits_count.index(max(splits_count))]]
                
            
    #         self.welds = welds
    #         # self.welds_by_instance = weldsInst
    #         # self.welds_by_type = weldsCls

    #     else:
    #         print("A sample must be processed through the network before welds can be extracted")
    #         self.welds = None

    def extract_welds(self):
        self.welds, self.welds_by_type, self.welds_by_instance = \
             self.extract_welds_from(self.predictions['cls'], self.predictions['inst'])

        # if self.labels['cls'] and self.labels['inst']:
        #     self.labeled_welds, self.labeled_welds_by_type, self.labeled_welds_by_instance = \
        #         self.extract_welds_from(self.labels['cls'], self.labels['inst'])

    def get_centroid(self, xyzpoint_arr):
        return np.mean(xyzpoint_arr, axis=0)

    def extract_centroids(self):
        self.weld_centroids = {}
        for i in self.welds.keys():
            self.weld_centroids[i] = self.get_centroid(self.welds[i])

        # if self.labeled_welds:
            
        #     self.labeled_weld_centroids = {}
        #     for i in self.labeled_welds.keys():
        #         self.labeled_weld_centroids[i] = self.get_centroid(self.labeled_welds[i])

    def draw(self, fig, draw_instance_seg=True):
        ax = fig.axes[0]
        ax.clear()
        ax.set_aspect("equal")
        full = self.terminal_points
        ax.scatter(full[:,0], full[:,1], c="beige")
        if draw_instance_seg:
            colors = ["#003366", "#336699", "#3366cc", "#003399", "#000099", "#0000cc"]
            for inst in self.welds.keys():
                weld = self.welds[inst]
                ax.scatter(weld[:, 0],weld[:, 1], c=colors[inst-1])
        else:
            colors = ["#cc0000", "#cc0099"]
            print("self.welds_by_type")
            print(self.welds_by_type.keys())
            for cls in self.welds_by_type.keys():
                weld = self.welds_by_type[cls]
                ax.scatter(weld[:, 0],weld[:, 1], c=colors[cls-1])
        
        # for idx in self.weld_centroids.keys():
        #     colors = ["#553366", "#886699", "#8866cc", "#553399", "#550099", "#5500cc"]
        #     c = self.weld_centroids[idx]
        #     ax.scatter(c[0], c[1], c=colors[idx-1])

        

if __name__ == "__main__":
    a = time.time()
    sample_scan_path = "./data/bin_file/photoneo1_a_ply5.bin"
    sample_label_path = "./data/gt/photoneo1_a_gt_5.txt"


    if os.path.isfile(sample_scan_path) and os.path.isfile(sample_label_path):
        b = time.time()
        print("Time to start: ", b-a)
        detector = Detector()
        res = detector.process_sample(sample_scan_path)
        # res = detector.process_sample(sample_scan_path, sample_label_path)

        detector.extract_welds()
        print(detector.welds)
    else:
        print("File Error: At least one scan/label path is not correctly spec'd")