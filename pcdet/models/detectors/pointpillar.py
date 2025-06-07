from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, *args):
        if type(args[0]) == dict:
            return self.forward_batch_dict(*args)
        else:
            return self.forward_onnx(*args)

    def forward_batch_dict(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
        
    def forward_onnx(self, voxel_features, voxel_num_points, voxel_coords, batch_size):
        vfe, map_to_bev_module, backbone_2d, dense_head = self.module_list

        pillar_features = vfe(voxel_features, voxel_num_points, voxel_coords)
        spatial_features = map_to_bev_module(pillar_features, voxel_coords)
        spatial_features_2d = backbone_2d(spatial_features)
        batch_cls_preds, batch_box_preds, cls_preds_normalized = dense_head(spatial_features_2d, batch_size)

        return batch_cls_preds, batch_box_preds, cls_preds_normalized

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
