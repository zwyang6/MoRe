from model.model_mal import network

def build_network(args):

    model = network(args,
    backbone=args.backbone,
    num_classes=args.num_classes,
    pretrained=args.pretrained,
    aux_layer=args.aux_layer
    )
    param_groups = model.get_param_groups()

    return model, param_groups