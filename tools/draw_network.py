import argparse
import os

import torch
import tensorwatch as tw
from fcos_core.config import cfg
from torch.autograd import Variable
from fcos_core.data import make_data_loader
# from torchviz import make_dot
# from fcos_core.solver import make_lr_scheduler
# from fcos_core.solver import make_optimizer
from fcos_core.modeling.detector import build_detection_model
from collections import namedtuple
from distutils.version import LooseVersion
from graphviz import Digraph
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter


Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    # output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)
    if isinstance(var, (tuple, list, set)):
        output_nodes = tuple(v.grad_fn for v in var)
    elif isinstance(var, dict):
        output_nodes = [v.grad_fn for k,v in var.items()]
    else:
        output_nodes = (var.grad_fn, )

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, (tuple, list)):
        for v in var:
            add_nodes(v.grad_fn)
    elif isinstance(var, dict):
        for k, v in var.items():
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

    resize_graph(dot)

    return dot


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument("--name", type=str, default="decouple_fpn")
    # parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN=('coco_2017_val',)
    cfg.SOLVER.IMS_PER_BATCH=2
    cfg.freeze()
    data_loader = make_data_loader(cfg, is_train=True, is_distributed=False)

    print("Start to draw network architecture...")
    model = build_detection_model(cfg)
    model = model.cuda()
    model.train()
    device = torch.device("cuda")
    
    for images,targets,_ in data_loader:
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        dot = make_dot(model(images, targets))
        dot.render(F"./{args.name}", view=True)
        break
    print("Finish.")

if __name__=='__main__':
    main()

