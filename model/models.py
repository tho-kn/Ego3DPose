def create_model(opt):
    print(opt.model)

    if opt.model == 'egoglass':
        from .egoglass_model import EgoGlassModel
        model = EgoGlassModel()

    elif opt.model == "ego3dpose_heatmap_shared":
        from .ego3dpose_heatmap_shared_model import Ego3DPoseHeatmapSharedModel
        model = Ego3DPoseHeatmapSharedModel()
        
    elif opt.model == 'ego3dpose_autoencoder':
        from .ego3dpose_autoencoder_model import Ego3DPoseAutoEncoderModel
        model = Ego3DPoseAutoEncoderModel()

    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model