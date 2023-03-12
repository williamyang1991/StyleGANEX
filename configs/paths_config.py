dataset_paths = {
    'ffhq': 'data/train/ffhq/realign320x320/',
    'ffhq_test': 'data/train/ffhq/realign320x320test/',
    'ffhq1280': 'data/train/ffhq/realign1280x1280/',
    'ffhq1280_test': 'data/train/ffhq/realign1280x1280test/',    
    'ffhq_train_sketch': 'data/train/ffhq/realign640x640sketch/',
    'ffhq_test_sketch': 'data/train/ffhq/realign640x640sketchtest/',
    'ffhq_train_segmentation': 'data/train/ffhq/realign320x320mask/',
    'ffhq_test_segmentation': 'data/train/ffhq/realign320x320masktest/',
    'toonify_in': 'data/train/pixar/trainA/',
    'toonify_out': 'data/train/pixar/trainB/',
    'toonify_test_in': 'data/train/pixar/testA/',
    'toonify_test_out': 'data/train/testB/',
}

model_paths = {
    'stylegan_ffhq': 'pretrained_models/stylegan2-ffhq-config-f.pt',
    'ir_se50': 'pretrained_models/model_ir_se50.pth',
    'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
    'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
    'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
    'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
