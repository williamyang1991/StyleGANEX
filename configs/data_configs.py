from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    'ffhq_encode': {
        'transforms': transforms_config.EncodeTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['ffhq_test'],
        'test_target_root': dataset_paths['ffhq_test'],
    },
    'ffhq_sketch_to_face': {
        'transforms': transforms_config.SketchToImageTransforms,
        'train_source_root': dataset_paths['ffhq_train_sketch'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['ffhq_test_sketch'],
        'test_target_root': dataset_paths['ffhq_test'],
    },
    'ffhq_seg_to_face': {
        'transforms': transforms_config.SegToImageTransforms,
        'train_source_root': dataset_paths['ffhq_train_segmentation'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['ffhq_test_segmentation'],
        'test_target_root': dataset_paths['ffhq_test'],
    },
    'ffhq_super_resolution': {
        'transforms': transforms_config.SuperResTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq1280'],
        'test_source_root': dataset_paths['ffhq_test'],
        'test_target_root': dataset_paths['ffhq1280_test'],
    },
    'toonify': {
        'transforms': transforms_config.ToonifyTransforms,
        'train_source_root': dataset_paths['toonify_in'],
        'train_target_root': dataset_paths['toonify_out'],
        'test_source_root': dataset_paths['toonify_test_in'],
        'test_target_root': dataset_paths['toonify_test_out'],
    },   
    'ffhq_edit': {
        'transforms': transforms_config.EditingTransforms,
        'train_source_root': dataset_paths['ffhq'],
        'train_target_root': dataset_paths['ffhq'],
        'test_source_root': dataset_paths['ffhq_test'],
        'test_target_root': dataset_paths['ffhq_test'],
    },
}
