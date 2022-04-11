import argparse

import torch
from torch_geometric.nn import Node2Vec
import os
from ogb.linkproppred import PygLinkPropPredDataset
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_sparse import coalesce, SparseTensor


def save_embedding(model, args):
    torch.save(model.embedding.weight.data.cpu(), os.path.join(args.res_dir, 'embedding.pt'))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--res_dir', type=str, default='log')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default=True)
    parser.add_argument('--train_on_subgraph', type=str2bool, default=True)
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--year', type=int, default=2010)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir, exist_ok=True)

    dataset = PygLinkPropPredDataset(name='ogbl-collab', root='dataset')
    data = dataset[0]

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()

    print(args)

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)

    if args.data_name == 'ogbl-collab':
        # only train edges after specific year
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

        # Use training + validation edges
        if args.use_valedges_as_input:
            full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
            full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
            # create adjacency matrix
            new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0],
                                      col=new_edge_index[1],
                                      value=new_edge_weight.to(torch.float32))
            data.edge_index = new_edge_index

            # if args.use_coalesce:
            #     full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

            # edge weight normalization
            split_edge['train']['edge'] = full_edge_index.t()
            deg = data.adj_t.sum(dim=1).to(torch.float)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            split_edge['train']['weight'] = deg_inv_sqrt[full_edge_index[0]] * full_edge_weight * deg_inv_sqrt[
                full_edge_index[1]]

        # reindex node ids on sub-graph
        if args.train_on_subgraph:
            # extract involved nodes
            row, col, edge_weight = data.adj_t.coo()
            subset = set(row.tolist()).union(set(col.tolist()))
            subset, _ = torch.sort(torch.tensor(list(subset)))
            # For unseen node we set its index as -1
            n_idx = torch.zeros(num_nodes, dtype=torch.long) - 1
            n_idx[subset] = torch.arange(subset.size(0))
            # Reindex edge_index, adj_t, num_nodes
            data.edge_index = n_idx[data.edge_index]
            data.adj_t = SparseTensor(row=n_idx[row], col=n_idx[col], value=edge_weight)
            num_nodes = subset.size(0)
            if hasattr(data, 'x'):
                if data.x is not None:
                    data.x = data.x[subset]
            # Reindex train valid test edges
            split_edge['train']['edge'] = n_idx[split_edge['train']['edge']]
            split_edge['valid']['edge'] = n_idx[split_edge['valid']['edge']]
            split_edge['valid']['edge_neg'] = n_idx[split_edge['valid']['edge_neg']]
            split_edge['test']['edge'] = n_idx[split_edge['test']['edge']]
            split_edge['test']['edge_neg'] = n_idx[split_edge['test']['edge_neg']]

    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model, args)
        save_embedding(model, args)


if __name__ == "__main__":
    main()
